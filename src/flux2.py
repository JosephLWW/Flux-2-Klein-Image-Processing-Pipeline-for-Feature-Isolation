#!/usr/bin/env python3
"""
===============================================================================
FLUX.2 Production Image Standardization Pipeline (Native Image Editing)
===============================================================================
Description:
    Production-ready pipeline using Hugging Face `diffusers` and PyTorch
    to standardize product images (~6GB dataset) using FLUX.2-dev.
    
    FLUX.2 integrates native structural and image-editing capabilities
    using its unified transformer and vision-language encoder (Mistral 24B),
    eliminating the need for ControlNet, external edge detectors (Canny), or
    classic SDEdit latent noising (strength).

Features:
    - Memory-safe for 40GB GPUs via sequential CPU offloading & expandable segments.
    - Flexible CSV matching & cleaning (handles missing data, extensions).
    - Native FLUX.2 Image-to-Image / Reference editing integration.
    - Resumable checkpointing (skips existing outputs).
    - Robust error logging (writes to generation_errors.log without crashing).
===============================================================================
"""

import os
import sys

# Configuración de asignación de memoria para evitar fragmentación en PyTorch CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

# HuggingFace Hub authentication
try:
    from huggingface_hub import login as hf_login, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Try importing diffusers components for FLUX.2
try:
    import diffusers
    from diffusers import (
        AutoPipelineForImage2Image,
        Flux2Pipeline,
    )
except ImportError as e:
    print(f"[!] Error importando componentes de diffusers: {e}")
    diffusers = None

from tqdm import tqdm


def setup_huggingface_auth(token: Optional[str] = None) -> bool:
    """
    Authenticates with HuggingFace Hub using (in order of priority):
      1. Token passed via --hf-token argument
      2. HF_TOKEN environment variable
      3. Cached token from ~/.cache/huggingface/token (set by hf auth login)
    Returns True if authentication succeeded, False otherwise.
    """
    if not HF_HUB_AVAILABLE:
        print("Warning: huggingface_hub not available. Skipping authentication.")
        return False

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
            print("    Run:  .venv/bin/hf auth login  or pass  --hf-token YOUR_TOKEN")
            return False
    return True


# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logger(log_file_path: Path) -> logging.Logger:
    """Configures file and console logging."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Flux2Standardizer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File Handler for error logging
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler for progress/info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# DATA PREPROCESSING CLASS
# =============================================================================
class DataPreprocessor:
    """
    Handles CSV loading, merging, cleaning, and input image path validation.
    """

    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = Path(base_dir)
        self.logger = logger

    def find_file(self, candidates: List[Path]) -> Optional[Path]:
        """Returns the first existing path from candidate locations."""
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_and_merge_metadata(
        self,
        mapping_csv_rel: str = "data/artikelnummer_to_image.csv",
        typicality_csv_rel: str = "data/article_typicality.csv",
    ) -> pd.DataFrame:
        """
        Loads mapping CSV and typicality CSV, cleans invalid entries,
        and merges on 'artikelnummer'.
        """
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

        if not mapping_path:
            raise FileNotFoundError(
                f"Mapping CSV not found at {self.base_dir / mapping_csv_rel} or alternative locations."
            )
        if not typicality_path:
            raise FileNotFoundError(
                f"Typicality CSV not found at {self.base_dir / typicality_csv_rel} or alternative locations."
            )

        print(f"Loading mapping CSV from: {mapping_path}")
        mapping_df = pd.read_csv(mapping_path)

        print(f"Loading typicality CSV from: {typicality_path}")
        typicality_df = pd.read_csv(typicality_path)

        if "artikelnummer" not in mapping_df.columns or "image_name" not in mapping_df.columns:
            raise ValueError(f"Mapping CSV must contain 'artikelnummer' and 'image_name'. Found: {mapping_df.columns.tolist()}")

        if "artikelnummer" not in typicality_df.columns:
            raise ValueError(f"Typicality CSV must contain 'artikelnummer'. Found: {typicality_df.columns.tolist()}")

        mapping_df["artikelnummer"] = mapping_df["artikelnummer"].astype(str).str.strip()
        typicality_df["artikelnummer"] = typicality_df["artikelnummer"].astype(str).str.strip()

        mapping_df = mapping_df.dropna(subset=["image_name", "artikelnummer"])
        mapping_df["image_name"] = mapping_df["image_name"].astype(str).str.strip()
        mapping_df = mapping_df[mapping_df["image_name"] != ""]

        def normalize_extension(filename: str) -> str:
            p = Path(filename)
            if not p.suffix:
                return f"{filename}.jpg"
            return filename

        mapping_df["image_name"] = mapping_df["image_name"].apply(normalize_extension)

        merged_df = pd.merge(mapping_df, typicality_df, on="artikelnummer", how="left")
        merged_df = merged_df.drop_duplicates(subset=["image_name"])

        print(f"Metadata merge successful. Total valid articles ready for processing: {len(merged_df)}")
        return merged_df

    def filter_existing_local_samples(
        self,
        df: pd.DataFrame,
        samples_dir_rel: str = "samples",
    ) -> pd.DataFrame:
        """
        Filters metadata DataFrame to ONLY include image files that actually exist
        locally in the samples directory.
        """
        samples_path = self.find_file([
            self.base_dir / samples_dir_rel,
            self.base_dir / "samples",
            self.base_dir / "Samples",
            Path(samples_dir_rel),
            Path("samples"),
            Path("Samples"),
        ])

        if not samples_path or not samples_path.exists():
            print(f"Warning: Samples directory '{samples_dir_rel}' not found. Returning original metadata.")
            return df

        local_files = {f.name for f in samples_path.glob("*") if f.is_file()}
        filtered_df = df[df["image_name"].isin(local_files)].copy()

        print(
            f"\n[+] Local Samples Filter Applied:"
            f"\n    Dataset reduced from {len(df)} total articles to "
            f"{len(filtered_df)} locally available image(s) in '{samples_path}'."
        )
        return filtered_df


# =============================================================================
# FLUX.2 MODEL INITIALIZATION & INFERENCE CLASS
# =============================================================================
class Flux2Standardizer:
    """
    Encapsulates FLUX.2 native reference/editing loading,
    memory optimizations, and image transformation logic.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-dev",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32

        print("\nInitializing FLUX.2 Native Editing Pipeline...")
        print(f"  Base Model:    '{self.model_id}'")
        print(f"  Target Device: {self.device} | Precision: {self.torch_dtype}")

        if diffusers is None:
            raise ImportError("The 'diffusers' library is not installed correctly or failed to import. Please check your environment.")

        print("Loading FLUX.2 weights...")
        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            )
        except Exception as e:
            print(f"Fallback to explicit Flux2Pipeline due to: {e}")
            self.pipe = Flux2Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            )

        # Apply VRAM Optimizations for 40GB / High-Memory Multi-Modal models
        if self.device == "cuda":
            if enable_cpu_offload:
                try:
                    # En GPUs de 40GB, sequential offload evita el OOM al cargar
                    # el encoder de texto multimodal Mistral capa por capa.
                    self.pipe.enable_sequential_cpu_offload()
                    print("  [+] VRAM Optimization: Sequential CPU Offloading ENABLED (layer-by-layer)")
                except AttributeError:
                    self.pipe.enable_model_cpu_offload()
                    print("  [+] VRAM Optimization: Model CPU Offloading ENABLED")
            else:
                self.pipe.to(self.device)

            if enable_vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
                print("  [+] VRAM Optimization: VAE Slicing ENABLED")
        else:
            self.pipe.to("cpu")

        print("FLUX.2 Pipeline ready.\n")

    def standardize_image(
        self,
        input_image: Image.Image,
        prompt: str,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        seed: int = 42,
    ) -> Image.Image:
        """
        Transforms input product image into a monochrome CAD/clay standard render
        using FLUX.2 native image editing and structural preservation.
        """
        image_rgb = input_image.convert("RGB")
        w, h = image_rgb.size
        target_w = (w // 16) * 16
        target_h = (h // 16) * 16
        if target_w != w or target_h != h:
            image_rgb = image_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Run FLUX.2 native image editing inference
        output = self.pipe(
            prompt=prompt,
            image=image_rgb,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        return output


# =============================================================================
# PIPELINE RUNNER & RESUME EXECUTION CLASS
# =============================================================================
class PipelineRunner:
    """
    Coordinates batch image loading, output directory checking (resume logic),
    FLUX.2 inference calls, progress bars, and exception logging.
    """

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
    ):
        self.base_dir = Path(base_dir)
        if input_dir_override:
            self.input_dir = (
                Path(input_dir_override)
                if Path(input_dir_override).is_absolute()
                else self.base_dir / input_dir_override
            )
        else:
            self.input_dir = self.resolve_directory(["images", "samples", "Samples", "."])
        self.output_dir = self.base_dir / "images_standardized"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.error_log_path = self.base_dir / "generation_errors.log"
        self.logger = setup_logger(self.error_log_path)

        self.standardizer = model_standardizer
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def resolve_directory(self, candidates: List[str]) -> Path:
        """Finds existing directory candidate containing image files."""
        for cand in candidates:
            p = self.base_dir / cand
            if p.exists() and p.is_dir() and any(f.is_file() for f in p.glob("*")):
                return p
            p_root = Path(cand)
            if p_root.exists() and p_root.is_dir() and any(f.is_file() for f in p_root.glob("*")):
                return p_root

        for cand in candidates:
            p = self.base_dir / cand
            if p.exists() and p.is_dir():
                return p
            p_root = Path(cand)
            if p_root.exists() and p_root.is_dir():
                return p_root

        default_p = self.base_dir / "images"
        default_p.mkdir(parents=True, exist_ok=True)
        return default_p

    def process_dataset(self, metadata_df: pd.DataFrame, max_samples: Optional[int] = None):
        """
        Iterates over items in dataset, skipping processed images, and handling errors.
        """
        print(f"\nInput images directory:  {self.input_dir}")
        print(f"Output images directory: {self.output_dir}")
        print(f"Error Log File:          {self.error_log_path}")

        records = metadata_df.to_dict("records")
        if max_samples:
            records = records[:max_samples]

        skipped_count = 0
        processed_count = 0
        failed_count = 0

        pbar = tqdm(records, desc="Standardizing Products", unit="img")

        for record in pbar:
            image_name = record["image_name"]
            output_file_path = self.output_dir / image_name

            # 1. Checkpoint & Resume Logic
            if output_file_path.exists() and output_file_path.stat().st_size > 0:
                skipped_count += 1
                pbar.set_postfix({"Processed": processed_count, "Skipped": skipped_count, "Failed": failed_count})
                continue

            input_file_path = self.input_dir / image_name
            if not input_file_path.exists():
                alt_paths = [
                    self.base_dir / "samples" / image_name,
                    self.base_dir / "Samples" / image_name,
                    self.base_dir / image_name,
                ]
                found_alt = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        input_file_path = alt_path
                        found_alt = True
                        break
                if not found_alt:
                    msg = f"Input file not found: {image_name}"
                    self.logger.warning(msg)
                    failed_count += 1
                    continue

            # 2. Inference wrapped in try/except block
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
                error_msg = f"Failed processing '{image_name}' (Artikel: {record.get('artikelnummer')}): {str(exc)}"
                self.logger.warning(error_msg, exc_info=True)
            finally:
                # Limpiar memoria residual tras cada imagen para evitar acumulación
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.set_postfix({"Processed": processed_count, "Skipped": skipped_count, "Failed": failed_count})

        print("\n===============================================================================")
        print("PIPELINE EXECUTION COMPLETED")
        print(f"  - Total Processed: {processed_count}")
        print(f"  - Skipped (Resume): {skipped_count}")
        print(f"  - Failed (Logged):  {failed_count}")
        print("===============================================================================\n")


# =============================================================================
# MAIN ENTRYPOINT & CLI ARGUMENTS
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2 Production Image Standardization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_data_dir = "./austria_data" if Path("./austria_data").exists() else "."

    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data_dir,
        help="Base working directory containing data/, images/, samples/",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Custom input images directory (e.g. 'samples' or 'images'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.2-dev",
        help="Hugging Face base model checkpoint ID for FLUX.2",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="[IGNORED IN FLUX.2] Kept for backward CLI compatibility. FLUX.2 uses native multimodal reference editing.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-Free Guidance (CFG) scale for FLUX.2 prompt adherence",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of denoising inference steps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline validation without initializing heavy neural network weights",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of images to process (useful for testing)",
    )
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Filter dataset to process ONLY images present in the local 'samples/' directory",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for accessing gated models.",
    )

    args = parser.parse_args()
    if args.strength is not None:
        print("[!] Advertencia: El parámetro '--strength' es ignorado por FLUX.2 en edición nativa.")

    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        base_dir = Path(".")

    print("===============================================================================")
    print("      FLUX.2 PRODUCT IMAGE STANDARDIZATION PIPELINE INITIALIZATION            ")
    print("===============================================================================")

    setup_huggingface_auth(token=args.hf_token)

    temp_logger = logging.getLogger("Init")

    # Step 1: Preprocessing & CSV Loading
    preprocessor = DataPreprocessor(base_dir=base_dir, logger=temp_logger)
    try:
        merged_metadata = preprocessor.load_and_merge_metadata()

        images_dir = base_dir / "images"
        images_has_files = images_dir.exists() and any(f.is_file() for f in images_dir.glob("*"))

        if args.samples_only or not images_has_files:
            merged_metadata = preprocessor.filter_existing_local_samples(merged_metadata)
            if args.input_dir is None:
                args.input_dir = "samples"
    except Exception as e:
        print(f"\n[ERROR] Metadata preprocessing failed: {e}")
        sys.exit(1)

    # Step 2: Model Loading (unless dry-run)
    model_standardizer = None
    if not args.dry_run:
        try:
            model_standardizer = Flux2Standardizer(
                model_id=args.model_id,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to load FLUX.2 model: {e}")
            print("Tip: If running on CPU or without GPU, use '--dry-run' to test data pipeline.")
            sys.exit(1)

    # Step 3: Run Batch Pipeline
    runner = PipelineRunner(
        base_dir=base_dir,
        model_standardizer=model_standardizer,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        input_dir_override=args.input_dir,
    )

    runner.process_dataset(metadata_df=merged_metadata, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
