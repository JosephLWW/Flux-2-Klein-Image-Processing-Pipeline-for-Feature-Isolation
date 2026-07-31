# Flux-2 Klein Image Processing Pipeline for Feature Isolation

A lightweight image-processing pipeline for isolating **basic visual structure** while suppressing non-essential surface details such as **textile texture, color, logos, and branding artifacts**.

## Overview

This project explores preprocessing techniques that reduce image complexity to core features (shape, edges, tonal structure) so downstream tasks can focus on fundamental form rather than style-specific or material-specific noise.

Typical use cases include:

- Feature extraction for classical computer vision
- Input normalization before model training/inference
- Removing visual confounders (fabric weave, print, logos)
- Building invariant representations of objects

## Objective

Given input images that may contain distracting surface properties, this pipeline aims to:

1. **Remove or suppress texture**
2. **Minimize color dependence**
3. **Reduce logo/print interference**
4. **Preserve structural cues** (silhouette, boundaries, major regions)

The output should retain essential geometric and intensity-driven information while discarding stylistic details.

## Repository Structure

```text
.
├── data/
│   ├── raw/                   # Input images
│   ├── interim/               # Intermediate outputs
│   └── processed/             # Final isolated-feature outputs
├── processed_examples/        # Examples of outputs by each model
│   ├── flux1+controlnet_sample.zip
│   ├── flux2_sample.zip
│   └── flux2klein_sample.zip
├── samples/                   # Sample Input Images
├── src/                       # Alternative pipelines for other models
│   ├── flux1_controlnet.py    # Alternative classical flux1 + controlnet pipeline
│   └── flux2.py               # Alternative FLUX.2 Image Standardization Pipeline
├── requirements.txt
└── README.md
```

## Installation

### 1) Clone repository

```bash
git clone https://github.com/JosephLWW/Flux-2-Klein-Image-Processing-Pipeline-for-Feature-Isolation.git
cd Flux-2-Klein-Image-Processing-Pipeline-for-Feature-Isolation
```

### 2) Create environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Notebook workflow

Open the notebook(s) and run step-by-step pre-processing:

```bash
jupyter lab
```

### Script workflow

```bash
python -m main.py
```

## Evaluation Ideas

To assess pipeline quality, consider:

- **Edge retention score** (how much meaningful contour remains)
- **Texture suppression ratio** (high-frequency attenuation)
- **Logo artifact reduction** (manual/automatic scoring)
- **Downstream task impact** (e.g., classification robustness)

## Roadmap

- [ ] Add reproducible benchmark dataset
- [ ] Add parameter sweep scripts
- [ ] Add objective quality metrics
- [ ] Add before/after report generation
- [ ] Package as installable module (`pip install -e .`)

## Contributing

Contributions are welcome. Suggested process:

1. Fork the repo
2. Create a feature branch
3. Add or improve pipeline components
4. Include visual before/after examples
5. Open a pull request
Specify your project license here (e.g., MIT, Apache-2.0, proprietary).

## Acknowledgments

- OpenCV / scikit-image / NumPy ecosystem
- Research and engineering work on structure-focused visual preprocessing
