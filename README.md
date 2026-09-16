# FLUX.2 Klein Production Image Standardization Pipeline

A lightweight, production-ready image-processing pipeline designed to standardize massive product image datasets (~6GB+) by isolating basic visual structures and suppressing non-essential surface details (e.g., textile texture, complex backgrounds, reflections, or logos).

Powered by **FLUX.2-klein-4B** via Hugging Face `diffusers` and optimized for High-Performance Computing (HPC) clusters.

## 📖 Overview & Objective

This project explores preprocessing techniques that reduce image complexity to core features (shape, edges, tonal structure) so downstream computer vision tasks can focus on fundamental form rather than style-specific or material-specific noise.

Given input images that may contain distracting surface properties, this pipeline aims to:
1. **Remove or suppress texture** (high-frequency attenuation)
2. **Minimize color dependence**
3. **Reduce logo/print interference**
4. **Preserve structural cues** (silhouette, boundaries, major regions)

## 🚀 Key Features

* **Multi-GPU Data Parallelism:** Natively scales across available GPUs (e.g., 4x H100) using `torch.multiprocessing`, drastically reducing inference time for large datasets.
* **HPC & SLURM Ready:** Includes a robust `.sh` batch script optimized for SLURM workload managers, with memory fragmentation mitigations (`expandable_segments:True`).
* **Direct ZIP Ingestion:** Capable of extracting and processing images directly from `.zip` archives (preventing filesystem strain on shared network drives) and re-compressing the standardized outputs.
* **Resumable Checkpointing:** Automatically detects existing valid outputs and skips them, allowing safe interruption and resumption of long-running jobs.

## 🧠 The Structural Prompt

The pipeline utilizes an `AutoPipelineForImage2Image` to transform input images based on a highly specific structural prompt designed to neutralize stylistic variance:

> *"CAD-style monochrome render of the exact object isolated on a pure flat white background, zero drop shadows, no floor shadow. Clean uniform matte light-grey surface, zero surface texture, zero patterns, zero logos, zero text, zero color. no mannequin, no display stand or shelf, only the original object. High geometric accuracy, sharp structural seams, clipping path isolated"*

## 📁 Repository Structure

```text
.
├── data/
│   ├── artikelnummer_to_image.csv   # Optional: Metadata mapping
│   └── raw_archives/                # Input zip files
├── images/                          # Extracted/Input raw images
├── images_standardized/             # Output directory for processed images
├── processed_examples/              # Examples of outputs (before/after)
├── samples/                         # Small subset for quick testing
├── src/                             # Alternative pipelines
│   ├── flux1_controlnet.py          # Classical flux1 + controlnet approach
│   └── flux2.py                     # Standard FLUX.2 approach
├── main.py                          # Core Python pipeline (Multi-GPU enabled)
├── run_flux2klein.sh                # SLURM batch script for HPC deployment
├── requirements.txt                 # Python dependencies
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/JosephLWW/Flux-2-Klein-Image-Processing-Pipeline-for-Feature-Isolation.git
cd Flux-2-Klein-Image-Processing-Pipeline-for-Feature-Isolation
```

### 2. Set up the Environment (Local or HPC)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Hugging Face Authentication
The pipeline requires access to the gated `black-forest-labs/FLUX.2-klein-4B` model.
1. Create a file named `token.txt` in the root directory containing your HF token.
2. Alternatively, export it as an environment variable: `export HF_TOKEN="your_token_here"`.

## 🖥️ Usage

### Local Execution (Single/Multi GPU)

**Process all images from a zip file (Default Behavior):**
By default, the script looks for `images.zip`, extracts valid images, processes them, and outputs `images_standardized.zip`.
```bash
python main.py --process-all-zip
```

**Process a local folder directly (No ZIP extraction):**
Useful if your images are already extracted in the `images/` directory.
```bash
python main.py --skip-zip
```

**Quick Test (Process only 5 samples):**
```bash
python main.py --samples-only --max-samples 5
```

### HPC Cluster Execution (SLURM)

To run the pipeline on an HPC cluster, edit the paths in `run_flux2klein.sh` to match your environment, then submit the job:
```bash
sbatch run_flux2klein.sh
```
Check the generated `flux_test_<JobID>.log` and `flux_test_<JobID>.err` files for progress and debugging.

## 📊 Evaluation & Roadmap

To assess pipeline quality in downstream tasks, we are evaluating:
- **Edge retention score:** How much meaningful contour remains.
- **Texture suppression ratio:** High-frequency attenuation.
- **Logo artifact reduction:** Manual/automatic scoring.

**Upcoming Features:**
- [ ] Add reproducible benchmark dataset.
- [ ] Add parameter sweep scripts.
- [ ] Package as installable module (`pip install -e .`).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/JosephLWW/Flux-2-Klein-Image-Processing-Pipeline-for-Feature-Isolation/issues).

## 📄 License & Acknowledgments
* Developed by [Joseph Wan](https://github.com/JosephLWW) (2026).
* Distributed under the MIT License.
* Powered by [Black Forest Labs FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and the Hugging Face `diffusers` ecosystem.
