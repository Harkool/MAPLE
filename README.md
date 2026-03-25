# MAPLE: Interpretable Deep Learning for Potent and Selective Antimicrobial Peptides

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)

**MAPLE (Multifunctional AMP Learning Engine)** is an interpretable deep learning framework for antimicrobial peptide modeling. It integrates **protein language model representations** with **knowledge-enhanced physicochemical encodings** to support both:

- **Binary AMP identification** (`AMP` task)
- **Multi-label functional prediction** over **14 AMP-related activity / phenotype labels** (`MTL` task)

This repository provides the reference implementation used in our manuscript, together with a reproducible pipeline for **feature caching, model training, threshold selection, and independent evaluation**.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Current Training Strategy](#current-training-strategy)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Data Layout](#data-layout)
- [Quickstart Demo](#quickstart-demo)
  - [AMP demo](#amp-demo)
  - [MTL demo](#mtl-demo)
- [Full Workflow](#full-workflow)
  - [1. Rebuild feature cache](#1-rebuild-feature-cache)
  - [2. Train models](#2-train-models)
  - [3. Evaluate on independent data](#3-evaluate-on-independent-data)
- [Outputs](#outputs)
- [Repository Structure](#repository-structure)
- [Interpretability Notes](#interpretability-notes)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

MAPLE adopts a **dual-stream representation framework** for peptide sequences:

- an **evolutionary stream** derived from pretrained or fallback sequence embeddings
- a **knowledge stream** derived from handcrafted physicochemical and sequence-level descriptors

These two streams are refined and fused to produce task-specific predictions.

### Tasks currently supported

MAPLE currently supports **two task-specific training settings**:

1. **AMP task**  
   Binary classification for **AMP vs non-AMP**

2. **MTL task**  
   Multi-label prediction over **14 functional / phenotype categories**, including efficacy-related and toxicity-related endpoints such as:
   - antibacterial
   - antifungal
   - anticancer
   - antiviral
   - hemolytic
   - cytotoxic  
   and other AMP-relevant labels

> **Important**  
> In the current codebase, these two tasks are trained as **separate runs**, each with its own checkpoint and threshold file.  
> They are **not** executed as a chained “Stage-1 then Stage-2” inference pipeline.

---

## Model Architecture

![MAPLE Model Architecture](Figure/architecture.jpg)

MAPLE uses a **dual-stream architecture**:

- **Evolutionary stream**  
  Processes sequence embeddings from the ESM-2 backend (or deterministic local fallback embeddings if `fair-esm` is unavailable)

- **Knowledge stream**  
  Processes knowledge-enhanced physicochemical sequence encodings

Both streams are refined by:

- **CARE** for local conserved / motif-aware feature extraction
- **ProBiMamba** for long-range dependency modeling

The refined representations are then fused via **CrossModalAttention** into a unified peptide representation, followed by a task-specific prediction head.

---

## Current Training Strategy

The current pipeline is organized as:

**feature cache → training/validation split → model selection on validation set → threshold export → independent evaluation**

### Key points

- `train.py` performs an internal **train/validation split** from the **Benchmark** dataset
- `Eval.py` is used for **post-training evaluation / inference**
- Feature generation can be precomputed via `build_feature_cache.py`
- Training produces:
  - a **task-specific best checkpoint**
  - a **task-specific threshold JSON file**
- Evaluation must use the **thresholds selected during training**
- Thresholds are **not re-tuned on the independent dataset**

This design avoids evaluation-time threshold leakage and keeps the independent set strictly for final reporting.

---

## Key Features

- Dual-stream fusion of:
  - **ESM-2 / embedding-based sequence representations**
  - **knowledge-enhanced physicochemical descriptors**
- Separate support for:
  - **binary AMP classification**
  - **14-label multi-label functional prediction**
- Interpretable architecture with motif-aware modules
- Feature caching for faster repeated training/evaluation
- Threshold export from training for reproducible evaluation
- Optional `fair-esm` backend with deterministic local fallback
- Memory-efficient training with support for gradient checkpointing

---

## Requirements

Recommended runtime: **Python 3.10+**

Install dependencies:

```bash
pip install -r requirements.txt
````

Optional pretrained ESM-2 backend:

```bash
pip install fair-esm
```

If `fair-esm` is not installed, the code falls back to **deterministic local embeddings**.

---

## Data Layout

This repository includes the following datasets:

### Benchmark datasets

Used by `train.py` for internal train/validation splitting.

* `Data/Benchmark/AMP.csv`
* `Data/Benchmark/MTL.csv`

### Independent datasets

Used only for final evaluation.

* `Data/Independent/AMP.csv`
* `Data/Independent/MTL.csv`

### Demo datasets

Small subsets for quick pipeline validation.

* `Data/Demo/AMP_demo.csv`
* `Data/Demo/MTL_demo.csv`

---

## Quickstart Demo

Use the demo data to quickly validate the full pipeline:

**build cache → train → evaluate**

> Demo runs are for **pipeline sanity check only** and should **not** be used for reporting model performance.

---

### AMP demo

#### 1) Build feature cache

```bash
python build_feature_cache.py \
  --data_csv Data/Demo/AMP_demo.csv \
  --label_cols label \
  --cache_dir cache_demo \
  --cache_name amp_demo \
  --overwrite
```

#### 2) Train

```bash
python train.py \
  --data_csv Data/Demo/AMP_demo.csv \
  --label_cols label \
  --epochs 1 \
  --save_dir runs_amp_demo \
  --cache_dir cache_demo \
  --cache_name amp_demo \
  --use_feature_cache \
  --strict_cache
```

#### 3) Evaluate

```bash
python Eval.py \
  --checkpoint runs_amp_demo/best_amp_model.pt \
  --threshold_file runs_amp_demo/best_amp_thresholds.json \
  --csv_path Data/Demo/AMP_demo.csv \
  --output_dir eval_amp_demo \
  --cache_dir cache_demo \
  --cache_name amp_demo \
  --use_feature_cache \
  --strict_cache
```

---

### MTL demo

#### 1) Build feature cache

```bash
python build_feature_cache.py \
  --data_csv Data/Demo/MTL_demo.csv \
  --label_cols anti_mammalian_cells antibacterial antibiofilm anticancer antifungal antigram-negative antigram-positive antihiv antimrsa antioxidant antiparasitic antiviral cytotoxic hemolytic \
  --cache_dir cache_demo \
  --cache_name mtl_demo \
  --overwrite
```

#### 2) Train

```bash
python train.py \
  --data_csv Data/Demo/MTL_demo.csv \
  --label_cols anti_mammalian_cells antibacterial antibiofilm anticancer antifungal antigram-negative antigram-positive antihiv antimrsa antioxidant antiparasitic antiviral cytotoxic hemolytic \
  --epochs 1 \
  --save_dir runs_mtl_demo \
  --cache_dir cache_demo \
  --cache_name mtl_demo \
  --use_feature_cache \
  --strict_cache
```

#### 3) Evaluate

```bash
python Eval.py \
  --checkpoint runs_mtl_demo/best_multilabel_model.pt \
  --threshold_file runs_mtl_demo/best_multilabel_thresholds.json \
  --csv_path Data/Demo/MTL_demo.csv \
  --output_dir eval_mtl_demo \
  --cache_dir cache_demo \
  --cache_name mtl_demo \
  --use_feature_cache \
  --strict_cache
```

---

## Full Workflow

## 1. Rebuild feature cache

Use **separate cache names** for benchmark and independent datasets.

### AMP cache

```bash
python build_feature_cache.py \
  --data_csv Data/Benchmark/AMP.csv \
  --label_cols label \
  --cache_dir cacheruns_amp_raw \
  --cache_name amp_benchmark \
  --overwrite
```

```bash
python build_feature_cache.py \
  --data_csv Data/Independent/AMP.csv \
  --label_cols label \
  --cache_dir cacheruns_amp_raw \
  --cache_name amp_independent \
  --overwrite
```

### MTL cache

```bash
python build_feature_cache.py \
  --data_csv Data/Benchmark/MTL.csv \
  --label_cols anti_mammalian_cells antibacterial antibiofilm anticancer antifungal antigram-negative antigram-positive antihiv antimrsa antioxidant antiparasitic antiviral cytotoxic hemolytic \
  --cache_dir cacheruns_amp_raw \
  --cache_name mtl_benchmark \
  --overwrite
```

```bash
python build_feature_cache.py \
  --data_csv Data/Independent/MTL.csv \
  --label_cols anti_mammalian_cells antibacterial antibiofilm anticancer antifungal antigram-negative antigram-positive antihiv antimrsa antioxidant antiparasitic antiviral cytotoxic hemolytic \
  --cache_dir cacheruns_amp_raw \
  --cache_name mtl_independent \
  --overwrite
```

---

## 2. Train models

`train.py` is the main training entry.

### AMP training

```bash
python train.py \
  --data_csv Data/Benchmark/AMP.csv \
  --label_cols label \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --epochs 100 \
  --val_ratio 0.2 \
  --seed 42 \
  --save_dir runs_amp \
  --cache_dir cacheruns_amp_raw \
  --cache_name amp_benchmark \
  --use_feature_cache \
  --strict_cache
```

### MTL training

```bash
python train.py \
  --data_csv Data/Benchmark/MTL.csv \
  --label_cols anti_mammalian_cells antibacterial antibiofilm anticancer antifungal antigram-negative antigram-positive antihiv antimrsa antioxidant antiparasitic antiviral cytotoxic hemolytic \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --epochs 100 \
  --val_ratio 0.2 \
  --seed 42 \
  --save_dir runs_mtl \
  --cache_dir cacheruns_amp_raw \
  --cache_name mtl_benchmark \
  --use_feature_cache \
  --strict_cache
```

---

## 3. Evaluate on independent data

`Eval.py` is the evaluation / inference entry.

### AMP evaluation

```bash
python Eval.py \
  --checkpoint runs_amp/best_amp_model.pt \
  --threshold_file runs_amp/best_amp_thresholds.json \
  --csv_path Data/Independent/AMP.csv \
  --output_dir eval_amp \
  --cache_dir cacheruns_amp_raw \
  --cache_name amp_independent \
  --use_feature_cache \
  --strict_cache
```

### MTL evaluation

```bash
python Eval.py \
  --checkpoint runs_mtl/best_multilabel_model.pt \
  --threshold_file runs_mtl/best_multilabel_thresholds.json \
  --csv_path Data/Independent/MTL.csv \
  --output_dir eval_mtl \
  --cache_dir cacheruns_amp_raw \
  --cache_name mtl_independent \
  --use_feature_cache \
  --strict_cache
```

---

## Outputs

Training writes task-specific artifacts.

### AMP task

* Best checkpoint: `best_amp_model.pt`
* Threshold file: `best_amp_thresholds.json`

### MTL task

* Best checkpoint: `best_multilabel_model.pt`
* Threshold file: `best_multilabel_thresholds.json`

### General notes

* Use a **unique `save_dir`** for each training run
* Use a **consistent `cache_name`** between cache building and later loading
* Evaluation should always use the **training-derived threshold file**
* Independent datasets are intended for **final evaluation only**

---

## Repository Structure

* `Data/` — benchmark, independent, and demo datasets
* `Figure/` — manuscript figures and architecture schematics
* `Module/` — core model modules (CARE, ProBiMamba, attention blocks, etc.)
* `build_feature_cache.py` — precompute and store feature caches
* `data.py` — dataset and dataloader utilities
* `loss.py` — task loss functions / imbalance-aware objectives
* `model.py` — MAPLE model definition
* `train.py` — training entry
* `Eval.py` — evaluation / inference entry

---


## Citation

If you use this code, please cite the manuscript (update as needed):

```bibtex
@article{submitted,
  title={MAPLE: Interpretable deep learning identifies selective antimicrobial peptides using joint evolutionary-physicochemical analysis},
  author={Liu, Hao and Shi, Yi and Guo, Feiyu and others},
  journal={Manuscript submitted},
  year={2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

Built with ❤️ by the GuoYu team | China Pharmaceutical University & Nanjing University

```
