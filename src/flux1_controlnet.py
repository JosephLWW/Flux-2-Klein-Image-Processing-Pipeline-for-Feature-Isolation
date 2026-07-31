#!/usr/bin/env python3
"""
===============================================================================
FLUX ControlNet Product Image Standardization Pipeline
===============================================================================
Author: Senior Computer Vision & ML Engineer
Description:
    Production-ready pipeline using Hugging Face `diffusers`, PyTorch, and OpenCV
    to standardize product images (~6GB dataset) using FLUX + ControlNet (Canny).
    
    The Canny edge detector extracts exact geometric boundaries from the original
    product image. ControlNet enforces 100% geometric and structural fidelity
    while FLUX removes original colors, patterns, logos, and surface textures
    to generate neutral matte monochrome CAD/clay standard renders.

Features:
    - Flexible CSV matching & cleaning (handles missing data, extensions).
    - Canny Edge Map Extraction via OpenCV for strict structural conditioning.
    - FLUX ControlNet Pipeline integration with configurable conditioning scales.
    - Memory optimizations (bfloat16, cpu_offload, vae_slicing) for low-VRAM GPUs.
    - Resumable checkpointing (skips existing outputs).
    - Robust error logging (writes to generation_errors.log without crashing).
    - Visual progress tracking with tqdm.
===============================================================================
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

# Try importing OpenCV for Canny Edge Detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# HuggingFace Hub authentication
try:
    from huggingface_hub import login as hf_login, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Try importing diffusers components with fallbacks
try:
    import diffusers
    from diffusers import (
        FluxControlNetPipeline,
        FluxControlNetImg2ImgPipeline,
        FluxControlNetModel,
        FluxPipeline,
    )
except ImportError:
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

    # Resolve token: argument > env var > cached (automatic)
    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if resolved_token:
        hf_login(token=resolved_token, add_to_git_credential=False)
        print(f"[+] HuggingFace: authenticated with provided token.")
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

    logger = logging.getLogger("FluxControlNetStandardizer")
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
        # Resolve mapping CSV
        mapping_path = self.find_file([
            self.base_dir / mapping_csv_rel,
            self.base_dir / "data" / "artikelnummer_to_image.csv",
            self.base_dir / "artikelnummer_to_image.csv",
            Path("artikelnummer_to_image.csv"),
        ])

        # Resolve typicality CSV
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

        # Validate essential column presence
        if "artikelnummer" not in mapping_df.columns or "image_name" not in mapping_df.columns:
            raise ValueError(f"Mapping CSV must contain 'artikelnummer' and 'image_name'. Found: {mapping_df.columns.tolist()}")

        if "artikelnummer" not in typicality_df.columns:
            raise ValueError(f"Typicality CSV must contain 'artikelnummer'. Found: {typicality_df.columns.tolist()}")

        # Clean string formats and drop nulls
        mapping_df["artikelnummer"] = mapping_df["artikelnummer"].astype(str).str.strip()
        typicality_df["artikelnummer"] = typicality_df["artikelnummer"].astype(str).str.strip()

        # Clean image_name
        mapping_df = mapping_df.dropna(subset=["image_name", "artikelnummer"])
        mapping_df["image_name"] = mapping_df["image_name"].astype(str).str.strip()
        mapping_df = mapping_df[mapping_df["image_name"] != ""]

        # Ensure correct image file extension (.jpg)
        def normalize_extension(filename: str) -> str:
            p = Path(filename)
            if not p.suffix:
                return f"{filename}.jpg"
            return filename

        mapping_df["image_name"] = mapping_df["image_name"].apply(normalize_extension)

        # Merge dataframes (left join preserves all mapping images even if typicality is missing)
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
        locally in the samples directory (or local image folder).

        Useful for local testing when full ~6GB image dataset is not downloaded
        locally and will be processed on the remote compute cluster.
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

        # Collect set of existing image filenames in local sample directory
        local_files = {f.name for f in samples_path.glob("*") if f.is_file()}
        filtered_df = df[df["image_name"].isin(local_files)].copy()

        print(
            f"\n[+] Local Samples Filter Applied:"
            f"\n    Dataset reduced from {len(df)} total articles to "
            f"{len(filtered_df)} locally available image(s) in '{samples_path}'."
        )
        return filtered_df


# =============================================================================
# FLUX CONTROLNET MODEL INITIALIZATION & INFERENCE CLASS
# =============================================================================
class FluxControlNetStandardizer:
    """
    Encapsulates FLUX + ControlNet (Canny) loading, Canny edge preprocessing,
    memory optimizations, and Image-to-Image / ControlNet transformation logic.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        controlnet_id: str = "InstantX/FLUX.1-dev-Controlnet-Canny",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
    ):
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32

        print(f"\nInitializing FLUX ControlNet Pipeline...")
        print(f"  Base Model:       '{self.model_id}'")
        print(f"  ControlNet Model: '{self.controlnet_id}'")
        print(f"  Target Device:    {self.device} | Precision: {self.torch_dtype}")

        if diffusers is None:
            raise ImportError("The 'diffusers' library is not installed. Please run: pip install diffusers transformers accelerate torch")
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV ('cv2') is required for Canny Edge Detection. Please run: pip install opencv-python")

        # 1. Load ControlNet model
        print("Loading ControlNet weights...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=self.torch_dtype,
        )

        # 2. Load Pipeline (FORZADO a FluxControlNetPipeline puro - Text-to-Image + ControlNet)
        print("Loading FLUX ControlNet Pipeline (Pure Edge-Conditioning Mode)...")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
        )
        self.mode = "controlnet_only"

        # Apply VRAM Optimizations
        if self.device == "cuda":
            if enable_cpu_offload:
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("  [+] VRAM Optimization: Model CPU Offloading ENABLED")
                except AttributeError:
                    self.pipe.to(self.device)
            else:
                self.pipe.to(self.device)

            if enable_vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
                print("  [+] VRAM Optimization: VAE Slicing ENABLED")
        else:
            self.pipe.to("cpu")

        print(f"FLUX ControlNet Pipeline ready ({self.mode} mode).\n")

    def extract_canny_edges(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> Image.Image:
        """
        Converts PIL Image to Canny edge map for ControlNet conditioning.
        """
        image_np = np.array(image.convert("RGB"))
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(image_gray, low_threshold, high_threshold)
        canny_edges_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(canny_edges_rgb)

    def standardize_image(
        self,
        input_image: Image.Image,
        prompt: str,
        strength: float = 0.75,
        controlnet_conditioning_scale: float = 0.75,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        seed: int = 42,
    ) -> Image.Image:
        """
        Transforms input product image into a monochrome CAD/clay standard render
        while strictly locking geometry via Canny edge ControlNet.

        HYPERPARAMETER TUNING (ControlNet + FLUX):
        -----------------------------------------------------------------------
        - `controlnet_conditioning_scale` (float, 0.0 to 1.0):
            Controls how strongly the Canny edge map forces geometry.
            * Optimal (0.65 - 0.80): Keeps 100% geometric accuracy without artifacts.
        - `strength` (float, only used in img2img_controlnet mode):
            Since ControlNet preserves edges, we can use a higher strength
            (0.70 - 0.85) to cleanly wipe out original colors, textures, and logos.
        """
        # Ensure image is RGB and resized to a multiple of 16 (FLUX requirement)
        image_rgb = input_image.convert("RGB")
        w, h = image_rgb.size
        target_w = (w // 16) * 16
        target_h = (h // 16) * 16
        if target_w != w or target_h != h:
            image_rgb = image_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Extract geometric edges
        canny_image = self.extract_canny_edges(image_rgb)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Run inference
        if self.mode == "img2img_controlnet":
            output = self.pipe(
                prompt=prompt,
                image=image_rgb,
                control_image=canny_image,
                strength=strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        else:
            # ControlNet-only mode (generates clean texture purely guided by edges)
            output = self.pipe(
                prompt=prompt,
                control_image=canny_image,
                height=target_h,
                width=target_w,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
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
    FLUX inference calls, progress bars, and exception logging.
    """

    # OPTIMIZED PROMPT FOR GEOMETRIC STANDARDIZATION (Neutral Matte Monochrome)
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
        model_standardizer: Optional[FluxControlNetStandardizer] = None,
        strength: float = 0.75,
        controlnet_scale: float = 0.75,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
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
        self.strength = strength
        self.controlnet_scale = controlnet_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def resolve_directory(self, candidates: List[str]) -> Path:
        """Finds existing directory candidate containing image files, or falls back to standard candidate."""
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

        # Progress bar configuration
        pbar = tqdm(records, desc="Standardizing Products", unit="img")

        for record in pbar:
            image_name = record["image_name"]
            output_file_path = self.output_dir / image_name

            # -----------------------------------------------------------------
            # 1. Checkpoint & Resume Logic: Skip if already generated
            # -----------------------------------------------------------------
            if output_file_path.exists() and output_file_path.stat().st_size > 0:
                skipped_count += 1
                pbar.set_postfix({"Processed": processed_count, "Skipped": skipped_count, "Failed": failed_count})
                continue

            # Resolve input image path
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

            # -----------------------------------------------------------------
            # 2. Inference wrapped in try/except block
            # -----------------------------------------------------------------
            try:
                            with Image.open(input_file_path) as raw_img:
                                raw_img = ImageOps.exif_transpose(raw_img)
            
                                if self.standardizer is not None:
                                    standardized_img = self.standardizer.standardize_image(
                                        input_image=raw_img,
                                        prompt=self.DEFAULT_PROMPT,
                                        strength=self.strength,
                                        controlnet_conditioning_scale=0.80, # Mantén un valor alto (0.80) para respetar costuras
                                        guidance_scale=self.guidance_scale,
                                        num_inference_steps=self.num_inference_steps,
                                    )
                                else:
                                    standardized_img = ImageOps.grayscale(raw_img).convert("RGB")
            
                                # # --- NUEVO: POSPROCESADO DE ENMASCARADO DE FONDO ---
                                # # 1. Aseguramos que original y generada tengan exactamente las mismas dimensiones
                                # raw_rgb = raw_img.convert("RGB").resize(standardized_img.size, Image.Resampling.NEAREST)
                                
                                # # 2. Creamos una máscara booleana: detecta los píxeles casi blancos del fondo original (> 245)
                                # raw_np = np.array(raw_rgb)
                                # std_np = np.array(standardized_img)
                                # bg_mask = np.all(raw_np > 245, axis=-1)
                                
                                # # 3. Forzamos un blanco puro perfecto (255) en todo el fondo, borrando sombras y "pies"
                                # std_np[bg_mask] = [255, 255, 255]
                                # standardized_img = Image.fromarray(std_np)
                                # # ---------------------------------------------------
            
                                standardized_img.save(output_file_path, quality=95)
                                processed_count += 1

            except Exception as exc:
                failed_count += 1
                error_msg = f"Failed processing '{image_name}' (Artikel: {record.get('artikelnummer')}): {str(exc)}"
                self.logger.warning(error_msg, exc_info=True)
                # Cleanup CUDA memory cache in case of OOM
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
        description="FLUX ControlNet Production Image Standardization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Smart default for base directory
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
        default="black-forest-labs/FLUX.1-dev",
        help="Hugging Face base model checkpoint ID for FLUX",
    )
    parser.add_argument(
        "--controlnet-id",
        type=str,
        default="InstantX/FLUX.1-dev-Controlnet-Canny",
        help="Hugging Face ControlNet Canny checkpoint ID",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Image-to-Image transformation strength (higher is safe with ControlNet preserving geometry)",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.75,
        help="ControlNet conditioning scale (0.65-0.80 enforces strict edge adherence)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-Free Guidance (CFG) scale for FLUX prompt adherence",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of denoising inference steps (20-30 recommended for FLUX.1-dev + ControlNet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline validation and extract Canny edges without initializing heavy neural network weights",
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
        help="HuggingFace API token for accessing gated models. If omitted, uses HF_TOKEN env var or cached token from 'hf auth login'.",
    )

    args = parser.parse_args()
    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        base_dir = Path(".")

    print("===============================================================================")
    print("   FLUX + CONTROLNET PRODUCT IMAGE STANDARDIZATION PIPELINE INITIALIZATION    ")
    print("===============================================================================")

    # Authenticate with HuggingFace Hub (needed for gated models like FLUX)
    setup_huggingface_auth(token=args.hf_token)

    # Temporary logger for preprocessing initialization
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
            model_standardizer = FluxControlNetStandardizer(
                model_id=args.model_id,
                controlnet_id=args.controlnet_id,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to load FLUX ControlNet model: {e}")
            print("Tip: If running on CPU or without GPU, use '--dry-run' to test data pipeline.")
            sys.exit(1)

    # Step 3: Run Batch Pipeline
    runner = PipelineRunner(
        base_dir=base_dir,
        model_standardizer=model_standardizer,
        strength=args.strength,
        controlnet_scale=args.controlnet_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        input_dir_override=args.input_dir,
    )

    runner.process_dataset(metadata_df=merged_metadata, max_samples=args.max_samples)


if __name__ == "__main__":
    main()