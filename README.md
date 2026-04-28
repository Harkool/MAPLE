# MAPLE: Interpretable Deep Learning for Selective Antimicrobial Peptide Prediction

MAPLE predicts AMP identity and 14 AMP-related functional activities from peptide sequences using joint evolutionary and physicochemical representations.

## Overview

MAPLE is an interpretable dual-stream framework for antimicrobial peptide prediction and functional profiling. It combines ESM-2 embeddings with knowledge-based physicochemical features and supports AMP identification, 14-category functional prediction, selectivity-oriented prioritization, and basic sequence property profiling.

This repository includes:

- local Streamlit inference
- command-line batch prediction
- evaluation scripts
- training scripts
- constrained fine-tuning utilities
- benchmark and independent processed datasets

## Architecture overview

![MAPLE architecture](Figure/architecture.jpg)

## Main features

- AMP identification from peptide sequences
- Functional prediction across 14 AMP-related categories
- Selectivity-oriented prioritization using antibacterial and hemolysis predictions
- Basic physicochemical descriptor calculation
- Optional motif-level interpretation
- Command-line inference
- Streamlit-based local prediction interface

The 14 functional categories are:

`anti_mammalian_cells`, `antibacterial`, `antibiofilm`, `anticancer`, `antifungal`, `antigram_negative`, `antigram_positive`, `antihiv`, `antimrsa`, `antioxidant`, `antiparasitic`, `antiviral`, `cytotoxic`, `hemolytic`

Note: internal code variables use `antigram_negative` and `antigram_positive`, while some checkpoint and dataset file names use `antigram-negative` and `antigram-positive`.

## Repository structure

```text
MAPLE/
├── app.py                        # Streamlit app entry logic
├── run.py                        # Thin Streamlit launcher
├── web_core/                     # Streamlit helper modules
├── predict.py                    # Command-line inference from CSV
├── eval.py                       # Checkpoint evaluation
├── train.py                      # Model training
├── Generate_pkl.py               # Unified feature PKL generation
├── model.py                      # MAPLE model implementation
├── data.py                       # Dataset and collate utilities
├── loss.py                       # Loss functions
├── Module/                       # CARE, ProBiMamba, Fusion, knowledge transformer modules
├── Data/
│   ├── Benchmark/
│   ├── Independent/
│   └── motif_reference.csv       # Motif interpretation reference used by Streamlit
├── MAPLE_checkpoints/            # Current local checkpoint layout in this repository
└── Figure/architecture.jpg       # Architecture figure
```

## Installation

### Option 1: Conda

```bash
conda create -n maple python=3.10 -y
conda activate maple
pip install torch pandas numpy scikit-learn streamlit plotly tqdm fair-esm
```

### Option 2: Existing environment

```bash
pip install torch pandas numpy scikit-learn streamlit plotly tqdm fair-esm
```

## Requirements

- Python 3.10 recommended
- PyTorch
- CUDA-compatible GPU recommended, CPU supported for small examples
- fair-esm
- pandas
- numpy
- scikit-learn
- streamlit
- plotly
- tqdm

Large-batch inference with ESM-2 embeddings is faster on GPU. The Streamlit interface is intended for small to moderate batches.

## Pretrained checkpoints

The current repository snapshot includes a local checkpoint layout under [Model](./Model). The Streamlit interface also supports a release-style folder such as `MAPLE_checkpoints/`.

Current local layout:

```text
MAPLE_checkpoints/
├── AMP.pt
├── knowledge_transformer.pt
├── thresholds.json
└── label/
    ├── anti_mammalian_cells/anti_mammalian_cells.pt
    ├── antibacterial/antibacterial.pt
    ├── antibiofilm/antibiofilm.pt
    ├── anticancer/anticancer.pt
    ├── antifungal/antifungal.pt
    ├── antigram-negative/antigram-negative.pt
    ├── antigram-positive/antigram-positive.pt
    ├── antihiv/antihiv.pt
    ├── antimrsa/antimrsa.pt
    ├── antioxidant/antioxidant.pt
    ├── antiparasitic/antiparasitic.pt
    ├── antiviral/antiviral.pt
    ├── cytotoxic/cytotoxic.pt
    └── hemolytic/hemolytic.pt
```

The Streamlit app accepts a user-specified checkpoint folder and searches compatible label checkpoint layouts:

- `label/<label>.pt`
- `label/<label>/<label>.pt`
- `<label>.pt`

If checkpoints are missing or cannot be loaded, `run.py` still performs sequence validation, descriptor calculation, and table generation, but no model probabilities are generated. The application does not generate random or dummy predictions.

## Quick start: Streamlit local interface

```bash
streamlit run run.py
```

The interface supports:

- manual sequence input
- CSV upload
- FASTA upload
- AMP prediction
- 14-category functional prediction for AMP-positive sequences
- physicochemical descriptor calculation
- downloadable CSV and FASTA outputs

Functional activity prediction is performed only for sequences classified as AMP-positive by the AMP screening model. Sequences that do not pass AMP screening are reported with AMP probability and sequence-derived descriptors only.

By default, the interface suggests `MAPLE_checkpoints` as the model folder. In this repository snapshot, existing local checkpoints are stored under `Model/`. The app includes compatibility fallback logic for that local layout.

## Input formats

### CSV input

```csv
sequence_id,sequence
pep_001,KWKLFKKIGAVLKVL
pep_002,GIGKFLHSAKKFGKAFVGEIMNS
```

If the CSV does not contain a column named `sequence`, the Streamlit interface allows users to select the sequence column manually.

### FASTA input

```fasta
>pep_001
KWKLFKKIGAVLKVL
>pep_002
GIGKFLHSAKKFGKAFVGEIMNS
```

### Manual input

One sequence per line, or FASTA-like text.

## Command-line inference

`predict.py` performs batch inference from CSV and writes probability columns to a CSV file.

Example:

```bash
python predict.py \
  --input_csv Data/Demo/AMP_demo.csv \
  --output_csv outputs/demo_predictions.csv \
  --sequence_col sequence \
  --label_dir Model/label \
  --knowledge_transformer_ckpt Model/knowledge_transformer.pt \
  --device auto
```

If you want to predict with a single checkpoint produced by `train.py`, use `--checkpoint` instead of `--label_dir`.

## Output fields

The local Streamlit output includes:

- `sequence_id`
- `sequence`
- `clean_sequence`
- `valid`
- `invalid_reason`
- `length_warning`
- `P_AMP`
- `AMP_label`
- `P_antibacterial`
- `P_hemolytic`
- `selectivity_score`
- `priority_group`
- functional probabilities and binary labels
- physicochemical descriptors
- additional sequence descriptors

`selectivity_score` is defined as:

```text
P_antibacterial - P_hemolytic
```

`priority_group` uses the following categories:

- `high_priority_selective`
- `effective_but_toxicity_flagged`
- `low_antibacterial_potential`
- `intermediate_or_uncertain`
- `properties_only`
- `invalid_sequence`

The command-line `predict.py` output is narrower and currently writes:

- `sequence`
- `prob_<label>` columns for the requested label set

## Sequence validation and physicochemical profiling

MAPLE accepts sequences composed of the 20 standard amino acids:

```text
ACDEFGHIKLMNPQRSTVWY
```

Sequences containing `B`, `J`, `O`, `U`, `X`, `Z`, or other unsupported symbols are retained in the output but skipped for model prediction.

The Streamlit interface reports approximate sequence-derived descriptors including:

- length
- approximate molecular weight
- approximate net charge
- charge density
- mean Kyte-Doolittle hydrophobicity
- fraction of positively charged residues
- fraction of negatively charged residues
- fraction of hydrophobic residues
- fraction of polar residues
- fraction of aromatic residues
- fraction of glycine
- fraction of proline

These descriptors are approximate sequence-derived descriptors for interpretation only and are not experimental measurements.

## Model architecture

MAPLE integrates two complementary input streams:

1. ESM-2 residue-level embeddings
2. Knowledge-based physicochemical residue features

The two streams are processed through a multi-scale sequence encoder and fused for peptide-level prediction. In the current implementation, the main components include:

- an ESM-2 embedding branch
- a knowledge-enhanced transformer branch
- a residual Mamba-style encoder block
- a residue ScConv block
- cross-modal fusion
- an MLP classification head for AMP screening and functional prediction

## Training and evaluation

MAPLE is trained as independent binary classifiers for AMP identification and each functional category.

### Build feature PKL

```bash
python Generate_pkl.py \
  --input_csv Data/Benchmark/MTL/antifungal.csv \
  --output_pkl Data/Benchmark/MTL/antifungal.pkl \
  --sequence_col sequence \
  --label_cols label \
  --knowledge_transformer_ckpt Model/knowledge_transformer.pt \
  --device auto
```

### Train a classifier

```bash
python train.py \
  --data_pkl Data/Benchmark/MTL/antifungal.pkl \
  --label_cols label \
  --save_dir out_stl/antifungal \
  --gpu 0
```

### Evaluate a checkpoint

```bash
python eval.py \
  --checkpoint out_stl/antifungal/antifungal.pt \
  --data_pkl Data/Independent/MTL/antifungal.pkl \
  --label_cols label \
  --threshold 0.5 \
  --device auto \
  --output_dir eval_outputs/antifungal
```

For local repository checkpoints, you can also evaluate files under `Model/label/...`. For example:

```bash
python eval.py \
  --checkpoint Model/label/antifungal/antifungal.pt \
  --data_pkl Data/Independent/MTL/antifungal.pkl \
  --label_cols label \
  --threshold 0.5 \
  --device auto \
  --output_dir eval_outputs/antifungal_model
```

## Optional constrained fine-tuning

```bash
python constrained_finetune.py \
  --init_checkpoint out_stl/antifungal/antifungal.pt \
  --tune_data_pkl Data/Independent/MTL/antifungal.pkl \
  --benchmark_pkl Data/Benchmark/MTL/antifungal.pkl \
  --label_cols label \
  --save_dir out_stl/antifungal_constrained_ft \
  --output_name antifungal_constrained.pt \
  --max_samples 5120 \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 30 \
  --gpu 0 \
  --threshold 0.5 \
  --benchmark_min_f1 0.0 \
  --benchmark_stop_floor 0.70 \
  --min_delta 0.0 \
  --save_any_valid \
  --save_last_if_none
```

## Data availability

Processed benchmark, independent, and demo files are included under [Data](./Data).

This repository contains processed CSV and PKL files used for training, evaluation, and demo inference. Source-database provenance and downstream manuscript details should be described in the accompanying paper or supplementary material.

## Reproducibility notes

- The independent dataset is intended for final evaluation rather than threshold selection.
- Task-specific thresholds are stored in `thresholds.json`.
- The Streamlit interface does not generate random or dummy predictions.
- If checkpoints are missing, probability columns are reported as unavailable rather than fabricated.
- Predictions are computational estimates and require experimental validation.

## Citation

If you use MAPLE, please cite:

```text
Liu H, Shi Y, Guo F, Wang J, Li J, Wang G, Zhan D-C, Hao H, Yu G.
MAPLE: Interpretable deep learning identifies selective antimicrobial peptides
using joint evolutionary-physicochemical analysis.
```

If the manuscript is still under review, replace the citation with your preferred preprint or review-status wording.

## License

No license file is currently included in this repository snapshot. Add an explicit license before public release.

## Contact

For questions, please contact the corresponding project authors.
