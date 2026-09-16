#!/usr/bin/env python3
"""
===============================================================================
FLUX.2 Klein Production Image Standardization Pipeline (Native Image Editing)
===============================================================================
Author: Joseph Wan
Year: 2026
Repository: https://github.com/JosephLWW/Flux-2-Klein-Image-Processing-Pipeline-for-Feature-Isolation/
===============================================================================
Description:
    Production-ready pipeline using Hugging Face `diffusers` and PyTorch
    to standardize product images (~6GB dataset) using FLUX.2-klein.
    
    Features:
    - Multi-GPU Data Parallelism native via torch.multiprocessing.
    - Memory-safe for 40GB/80GB GPUs.
    - Flexible CSV matching & cleaning.
    - Resumable checkpointing (skips existing outputs).
===============================================================================
"""

import os
import sys
import zipfile
import shutil

# Configuración de asignación de memoria para evitar fragmentación en PyTorch CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import logging
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.multiprocessing as mp

# HuggingFace Hub authentication
try:
    from huggingface_hub import login as hf_login, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Try importing diffusers components for FLUX.2
try:
    import diffusers
    from diffusers import AutoPipelineForImage2Image, Flux2Pipeline
except ImportError as e:
    print(f"[!] Error importando componentes de diffusers: {e}")
    diffusers = None

from tqdm import tqdm


def setup_huggingface_auth(token: Optional[str] = None) -> bool:
    """Authenticates with HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        print("Warning: huggingface_hub not available. Skipping authentication.")
        return False

    token_file = Path("token.txt")
    if not token and token_file.exists():
        with open(token_file, "r") as f:
            token = f.read().strip()
        os.environ["HF_TOKEN"] = token

    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if resolved_token:
        hf_login(token=resolved_token, add_to_git_credential=False)
        print("[+] HuggingFace: authenticated with provided token.")
    else:
        try:
            api = HfApi()
            user = api.whoami()
            print(f"[+] HuggingFace: using cached token (user: {user.get('name', 'unknown')}).")
        except Exception:
            print("[!] HuggingFace: no token found. Gated models may fail.")
            return False
    return True


def setup_logger(log_file_path: Path, name: str = "Flux2Klein") -> logging.Logger:
    """Configures file and console logging."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


class DataPreprocessor:
    """Handles CSV loading, merging, cleaning, and input image path validation."""
    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = Path(base_dir)
        self.logger = logger

    def find_file(self, candidates: List[Path]) -> Optional[Path]:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_and_merge_metadata(
        self,
        mapping_csv_rel: str = "data/artikelnummer_to_image.csv",
        typicality_csv_rel: str = "data/article_typicality.csv",
    ) -> pd.DataFrame:
        mapping_path = self.find_file([
            self.base_dir / mapping_csv_rel,
            self.base_dir / "data" / "artikelnummer_to_image.csv",
            self.base_dir / "artikelnummer_to_image.csv",
            Path("artikelnummer_to_image.csv"),
        ])

        typicality_path = self.find_file([
            self.base_dir / typicality_csv_rel,
            self.base_dir / "data" / "article_typicality.csv",
            self.base_dir / "article_typicality.csv",
            Path("article_typicality.csv"),
        ])

        if not mapping_path or not typicality_path:
            raise FileNotFoundError("Mapping or Typicality CSV not found.")

        mapping_df = pd.read_csv(mapping_path)
        typicality_df = pd.read_csv(typicality_path)

        mapping_df["artikelnummer"] = mapping_df["artikelnummer"].astype(str).str.strip()
        typicality_df["artikelnummer"] = typicality_df["artikelnummer"].astype(str).str.strip()

        mapping_df = mapping_df.dropna(subset=["image_name", "artikelnummer"])
        mapping_df["image_name"] = mapping_df["image_name"].astype(str).str.strip()
        mapping_df = mapping_df[mapping_df["image_name"] != ""]

        def normalize_extension(filename: str) -> str:
            return filename if Path(filename).suffix else f"{filename}.jpg"

        mapping_df["image_name"] = mapping_df["image_name"].apply(normalize_extension)
        merged_df = pd.merge(mapping_df, typicality_df, on="artikelnummer", how="left")
        merged_df = merged_df.drop_duplicates(subset=["image_name"])

        return merged_df

    def filter_existing_local_samples(self, df: pd.DataFrame, samples_dir_rel: str = "samples") -> pd.DataFrame:
        samples_path = self.find_file([
            self.base_dir / samples_dir_rel,
            self.base_dir / "Samples",
            Path(samples_dir_rel),
        ])

        if not samples_path or not samples_path.exists():
            return df

        local_files = {f.name for f in samples_path.glob("*") if f.is_file()}
        return df[df["image_name"].isin(local_files)].copy()


class Flux2Standardizer:
    """Encapsulates FLUX.2 klein native reference/editing loading and memory optimization."""
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False, # Desactivado por defecto para H100
        enable_vae_slicing: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype if "cuda" in self.device else torch.float32

        if diffusers is None:
            raise ImportError("diffusers library is missing.")

        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype, use_safetensors=True
            )
        except Exception:
            self.pipe = Flux2Pipeline.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype, use_safetensors=True
            )

        if "cuda" in self.device:
            if enable_cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                # Directamente a la GPU especificada (Ideal para H100/A100)
                self.pipe.to(self.device)

            if enable_vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
        else:
            self.pipe.to("cpu")

    def standardize_image(
        self,
        input_image: Image.Image,
        prompt: str,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        seed: int = 42,
    ) -> Image.Image:
        image_rgb = input_image.convert("RGB")
        w, h = image_rgb.size
        target_w, target_h = (w // 16) * 16, (h // 16) * 16
        
        if target_w != w or target_h != h:
            image_rgb = image_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            image=image_rgb,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        return output


class PipelineRunner:
    """Coordinates batch image loading, output directory checking (resume logic), and FLUX.2 inference."""
    DEFAULT_PROMPT = (
        "CAD-style monochrome render of the exact object isolated on a pure flat white background, zero drop shadows, no floor shadow. "
        "Clean uniform matte light-grey surface, zero surface texture, "
        "zero patterns, zero logos, zero text, zero color. "
        "no mannequin, no display stand or shelf, only the original object. "
        "High geometric accuracy, sharp structural seams, clipping path isolated"
    )

    def __init__(
        self,
        base_dir: Path,
        model_standardizer: Optional[Flux2Standardizer] = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        input_dir_override: Optional[str] = None,
        worker_id: int = 0,
    ):
        self.base_dir = Path(base_dir)
        self.worker_id = worker_id
        
        if input_dir_override:
            p = Path(input_dir_override)
            self.input_dir = p if p.is_absolute() else self.base_dir / p
        else:
            self.input_dir = self.resolve_directory(["images", "samples", "Samples", "."])
            
        self.output_dir = self.base_dir / "images_standardized"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.error_log_path = self.base_dir / "generation_errors.log"
        self.logger = setup_logger(self.error_log_path, name=f"Worker-{worker_id}")

        self.standardizer = model_standardizer
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def resolve_directory(self, candidates: List[str]) -> Path:
        for cand in candidates:
            p = self.base_dir / cand
            if p.exists() and p.is_dir() and any(f.is_file() for f in p.glob("*")):
                return p
        default_p = self.base_dir / "images"
        default_p.mkdir(parents=True, exist_ok=True)
        return default_p

    def process_records(self, records: List[dict]):
        if not records:
            return

        skipped_count = processed_count = failed_count = 0
        pbar = tqdm(records, desc=f"GPU {self.worker_id}", position=self.worker_id, unit="img")

        for record in pbar:
            image_name = record["image_name"]
            output_file_path = self.output_dir / image_name

            if output_file_path.exists() and output_file_path.stat().st_size > 0:
                skipped_count += 1
                pbar.set_postfix({"OK": processed_count, "Skip": skipped_count, "Fail": failed_count})
                continue

            input_file_path = self.input_dir / image_name
            if not input_file_path.exists():
                alt_paths = [
                    self.base_dir / "samples" / image_name,
                    self.base_dir / "Samples" / image_name,
                    self.base_dir / image_name,
                ]
                input_file_path = next((ap for ap in alt_paths if ap.exists()), None)

            if not input_file_path:
                self.logger.warning(f"Input file not found: {image_name}")
                failed_count += 1
                continue

            try:
                with Image.open(input_file_path) as raw_img:
                    raw_img = ImageOps.exif_transpose(raw_img)

                    if self.standardizer is not None:
                        standardized_img = self.standardizer.standardize_image(
                            input_image=raw_img,
                            prompt=self.DEFAULT_PROMPT,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                        )
                    else:
                        standardized_img = ImageOps.grayscale(raw_img).convert("RGB")

                    standardized_img.save(output_file_path, quality=95)
                    processed_count += 1

            except Exception as exc:
                failed_count += 1
                self.logger.warning(f"Failed '{image_name}': {str(exc)}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.set_postfix({"OK": processed_count, "Skip": skipped_count, "Fail": failed_count})


# =============================================================================
# MULTIPROCESSING WORKER
# =============================================================================
def worker_process(rank: int, chunks: List[List[dict]], args, base_dir_str: str):
    """Function executed by each spawned process (per GPU)."""
    records_chunk = chunks[rank]
    if not records_chunk:
        return

    device = f"cuda:{rank}"
    
    try:
        # Inicializamos el modelo en la GPU asignada
        standardizer = Flux2Standardizer(
            model_id=args.model_id,
            device=device,
            enable_cpu_offload=False, # H100 tiene suficiente VRAM, el offload hace todo más lento
            enable_vae_slicing=True
        )
    except Exception as e:
        print(f"[Worker {rank}] Error cargando modelo: {e}")
        return

    runner = PipelineRunner(
        base_dir=Path(base_dir_str),
        model_standardizer=standardizer,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        input_dir_override=args.input_dir,
        worker_id=rank
    )
    
    runner.process_records(records_chunk)


# =============================================================================
# MAIN ENTRYPOINT & CLI ARGUMENTS
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein Image Standardization")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for authentication.")
    parser.add_argument("--data-dir", type=str, default=".", help="Base working directory")
    parser.add_argument("--input-dir", type=str, default=None, help="Custom input images directory")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-only", action="store_true")
    parser.add_argument("--skip-zip", action="store_true", help="Omitir el procesamiento de archivos ZIP y usar carpetas directamente.")
    parser.add_argument("--zip-file", type=str, default="images.zip", help="Nombre del archivo zip de entrada.")
    parser.add_argument("--output-zip", type=str, default="images_standardized.zip", help="Nombre del archivo zip de salida final.")
    parser.add_argument("--process-all-zip", action="store_true", help="Procesar todas las imágenes del ZIP ignorando el CSV.")

    args = parser.parse_args()
    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        base_dir = Path(".")

    print("===============================================================================")
    print("      FLUX.2 KLEIN PRODUCT IMAGE PIPELINE (MULTI-GPU ENABLED)                  ")
    print("===============================================================================")

    setup_huggingface_auth(token=args.hf_token)

    # 1. Load Metadata or Extract/Read All Zip Images
    # Modo folder-first: si no se pide zip, miramos la carpeta 'images' directamente.
    use_zip = not args.skip_zip
    
    zip_path = base_dir / args.zip_file if not Path(args.zip_file).is_absolute() else Path(args.zip_file)
    extracted_images_dir = base_dir / "images"
    
    if (use_zip and (args.process_all_zip or not (base_dir / "data/artikelnummer_to_image.csv").exists())) or args.skip_zip:
        if args.skip_zip:
            print("[+] Modo skip-zip: Buscando imágenes directamente en la carpeta 'images'...")
            if not extracted_images_dir.exists():
                print(f"[ERROR] La carpeta {extracted_images_dir} no existe.")
                sys.exit(1)
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            all_files = [f.name for f in extracted_images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            records = [{"image_name": filename} for filename in sorted(all_files)]
            print(f"[+] Se detectaron {len(records)} imágenes en la carpeta para procesar.")
        
        elif zip_path.exists():
            print(f"\n[+] Extrayendo todas las imágenes de {zip_path} a {extracted_images_dir}...")
            # ... (rest of the extraction logic)
            extracted_images_dir.mkdir(parents=True, exist_ok=True)
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            extracted = skipped = 0
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    if member.is_dir():
                        continue
                    # Quitar prefijo de primer nivel (ej. 'images/xxx.jpg' -> 'xxx.jpg').
                    # El zip trae todo bajo 'images/', y si hacemos extractall a
                    # 'images/' crearíamos 'images/images/'. Por eso aplanamos a basename.
                    raw_name = member.filename
                    clean_name = Path(raw_name).name
                    if not clean_name or Path(clean_name).suffix.lower() not in valid_exts:
                        continue
                    # Defensa contra zip-slip
                    target = extracted_images_dir / clean_name
                    try:
                        target.resolve().relative_to(extracted_images_dir.resolve())
                    except ValueError:
                        continue
                    if target.exists() and target.stat().st_size > 0:
                        skipped += 1
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted += 1
            print(f"[+] Extracción completa: {extracted} nuevas, {skipped} ya existían.")

            # Obtener todas las imágenes válidas extraídas (plano, sin subcarpetas)
            all_files = [f.name for f in extracted_images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            records = [{"image_name": filename} for filename in sorted(all_files)]
            print(f"[+] Se detectaron {len(records)} imágenes en el ZIP para procesar.")
        else:
            print(f"\n[ERROR] No se encontró el archivo ZIP en {zip_path}")
            sys.exit(1)
    else:
        preprocessor = DataPreprocessor(base_dir=base_dir, logger=logging.getLogger("Init"))
        try:
            merged_metadata = preprocessor.load_and_merge_metadata()
            if args.samples_only:
                merged_metadata = preprocessor.filter_existing_local_samples(merged_metadata)
                if args.input_dir is None:
                    args.input_dir = "samples"
        except Exception as e:
            print(f"\n[ERROR] Preprocessing failed: {e}")
            sys.exit(1)

        records = merged_metadata.to_dict("records")

    if args.max_samples:
        records = records[:args.max_samples]

    # 2. Distribute Workload
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1 and not args.dry_run:
        print(f"\n[+] MULTI-GPU DETECTADO: Repartiendo {len(records)} imágenes entre {num_gpus} GPUs...\n")
        
        # Dividir records en partes casi iguales
        chunk_size = math.ceil(len(records) / num_gpus)
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        
        # Lanzar procesos paralelos (uno por GPU)
        mp.spawn(worker_process, args=(chunks, args, str(base_dir)), nprocs=num_gpus)
        
    else:
        # Fallback a 1 sola GPU o CPU
        print(f"\n[+] Ejecución estándar (1 Dispositivo detectado o Dry-Run).")
        model_standardizer = None if args.dry_run else Flux2Standardizer(
            model_id=args.model_id, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        runner = PipelineRunner(
            base_dir=base_dir,
            model_standardizer=model_standardizer,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            input_dir_override=args.input_dir,
            worker_id=0
        )
        runner.process_records(records)

    print("\n===============================================================================")
    print("PIPELINE EXECUTION COMPLETED")
    print("===============================================================================\n")

    # Compresión final de las imágenes estandarizadas
    output_dir = base_dir / "images_standardized"
    output_zip_path = base_dir / args.output_zip
    if not args.skip_zip and output_dir.exists() and any(output_dir.iterdir()):
        print(f"[+] Comprimiendo el directorio {output_dir} en {output_zip_path}...")
        shutil.make_archive(str(output_zip_path.with_suffix('')), 'zip', output_dir)
        print(f"[+] Compresión completada: {output_zip_path}")
    else:
        print(f"[+] Imágenes estandarizadas guardadas en: {output_dir}")


if __name__ == "__main__":
    # Importante para PyTorch Multiprocessing en Linux
    mp.set_start_method('spawn', force=True)
    main()
