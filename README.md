# ForensicsDetective

PDF provenance detection via binary-to-image classification.

## Overview

This project identifies which application generated a PDF file — Microsoft Word, Google Docs, or Python/ReportLab — by converting the raw PDF binary into a grayscale image and classifying it with machine learning. Each PDF creation engine embeds structurally distinct patterns in its output at the binary level, making provenance detectable even after metadata has been stripped.

## Dataset

PNG images are pre-generated and ready to use. No PDF conversion is needed.

| Folder | Source | Count |
|--------|--------|-------|
| `word_pdfs_png/` | Microsoft Word export | 398 |
| `google_docs_pdfs_png/` | Google Docs download | 396 |
| `python_pdfs_png/` | Python/ReportLab | 100 |

All documents share the same Wikipedia article content across classes, so binary signatures reflect creation method rather than document content.

## Project Structure

```
ForensicsDetective/
├── data/
│   ├── original_images/        # see Dataset section below
│   └── augmented_images/       # output of src/augmentation.py
│       ├── word/
│       │   ├── original/
│       │   ├── gaussian_noise/
│       │   ├── jpeg_compression/
│       │   ├── downsample_dpi/
│       │   ├── random_crop/
│       │   └── reduce_bit_depth/
│       ├── google/
│       └── python/
├── src/
│   ├── augmentation.py
│   ├── classification.py
│   └── analysis.py
├── results/
│   ├── performance_metrics.csv
│   ├── full_metrics.csv
│   ├── robustness_summary.csv
│   ├── statistical_significance.csv
│   ├── predictions.csv
│   ├── confusion_matrices/
│   └── robustness_plots/
├── reports/
│   └── final_research_report.pdf
├── word_pdfs_png/              # original Word images (398)
├── google_docs_pdfs_png/       # original Google Docs images (396)
├── python_pdfs_png/            # original Python images (100)
├── requirements.txt
├── SETUP.md
└── README.md
```

> **Note:** The original PNG images are kept in the root-level `*_pdfs_png/` folders (pre-generated, not reproduced). `data/original_images/` is a placeholder per the submission structure requirement.

## Pipeline

### Step 1 — Augmentation

Applies five distortions to each original image, producing a 6× dataset (original + 5 augmented variants per image).

```bash
python src/augmentation.py
```

| Augmentation | Parameters | Simulates |
|---|---|---|
| Gaussian Noise | σ ∈ [5, 20] | Scanner noise |
| JPEG Compression | quality ∈ [20, 80] | Lossy re-encoding |
| DPI Downsampling | 300 → 150 or 72 DPI | Low-resolution scanning |
| Random Crop | 1–3% per border | Page alignment shift |
| Bit-Depth Reduction | 8-bit → 4-bit | Reduced display quality |

### Step 2 — Classification

Trains four classifiers on the original images only, then evaluates each on the original test set and all five augmentation conditions.

```bash
python src/classification.py
```

Classifiers: SVM (RBF), SGD (hinge), Random Forest (200 trees), XGBoost (200 rounds).

Outputs:
- `results/performance_metrics.csv` — accuracy and confusion matrix per classifier per condition
- `results/predictions.csv` — per-sample predictions for statistical testing

### Step 3 — Analysis

Computes precision/recall/F1, generates confusion matrix plots, runs McNemar significance tests, and produces robustness curves.

```bash
python src/analysis.py
```

Outputs:
- `results/full_metrics.csv` — precision, recall, F1 per class and macro
- `results/confusion_matrices/` — 24 heatmap PNGs (4 classifiers × 6 conditions)
- `results/robustness_plots/` — robustness curves and accuracy drop bar chart
- `results/statistical_significance.csv` — pairwise McNemar p-values



## Results Summary

Baseline accuracy on original test set:

| Classifier | Accuracy |
|---|---|
| XGBoost | 100.0% |
| SGD | 99.4% |
| Random Forest | 98.9% |
| SVM | 97.2% |

Largest accuracy drop per classifier (worst augmentation condition):

| Classifier | Worst Condition | Accuracy Drop |
|---|---|---|
| XGBoost | Random Crop | −52.5% |
| SVM | Random Crop / DPI Downsample | −41.9% |
| Random Forest | Random Crop | −40.8% |
| SGD | Random Crop | −40.2% |

## Setup

See [SETUP.md](SETUP.md) for environment setup instructions.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
