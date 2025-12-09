# MAPLE — Multi-source AMP Learning Encoder  

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)]()
[![Paper](https://img.shields.io/badge/Paper-bioRxiv-red.svg)]()

> **MAPLE: Integrating ESM-2 embeddings with adaptive dual-stream architecture for accurate identification and multi-label functional prediction of antimicrobial peptides**


## 🌟 Highlights

- **Dual-Stream Fusion**: Integrates **ESM-2 embeddings** with **knowledge-based features** through parallel pathways
- **CARE Module**: Conservative Adaptive Representation Encoder extracts conserved local motifs via multi-scale convolutions with adaptive channel attention
- **ProBiMamba**: Efficient bidirectional selective state-space model captures long-range dependencies with linear complexity
- **Cross-Modal Attention**: Aligns and fuses heterogeneous feature representations for comprehensive sequence understanding

## 📋 Table of Contents

- [Background](#-background)
- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Customization](#-customization)
- [Citation](#-citation)
- [License](#-license)

## 🔬 Background

### The AMR Crisis

Antimicrobial resistance (AMR) poses a critical global health threat, projected to cause:
- **10 million deaths annually by 2050**
- **Nearly $100 trillion in economic losses**

### Why Antimicrobial Peptides?

AMPs offer unique advantages as next-generation therapeutics:
- **Broad-spectrum activity** through membrane-disruptive mechanisms
- **Rapid bactericidal action** against multidrug-resistant pathogens
- **Low resistance development** due to multi-target mechanisms
- **Multifunctional properties** beyond antimicrobial activity

### Computational Challenges

Traditional ML and recent deep learning approaches face limitations:
1. **Single-modality bias**: Over-reliance on either data-driven (PLMs) or knowledge-based features
2. **Simple fusion strategies**: Concatenation/addition fails to capture semantic complementarity
3. **Architectural constraints**: RNNs suffer from vanishing gradients; CNNs lack global context
4. **Limited interpretability**: Black-box models hinder biological insight
5. **Single-task focus**: Most models target only binary classification, ignoring functional diversity

### MAPLE's Innovation

MAPLE addresses these challenges through:
- **Multi-source integration**: Combines ESM-2's evolutionary knowledge with explicit physicochemical features
- **Adaptive dual-stream design**: Separate pathways optimized for different feature types
- **Efficient long-range modeling**: ProBiMamba replaces RNNs with linear-complexity SSMs
- **Cross-modal fusion**: Attention mechanisms align heterogeneous representations
- **Multi-task capability**: Simultaneous binary classification and 14-way functional annotation

## 🛠 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for ESM-2 15B)
- ~60GB disk space for model weights and datasets

### Environment Setup

```bash
# Create and activate conda environment
conda create -n maple python=3.10
conda activate maple

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Install ESM-2 (supports up to 15B parameters)
pip install "esm>=2.0" fair-esm biotite
```

### Verify Installation

```bash
python -c "import torch; import esm; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.2.0+cu121
CUDA available: True
```

## 📁 Repository Structure

```
MAPLE/
├── data/
│   └── Benchmark/
│       ├── AMP/                  # Binary classification (25,507 AMPs + 72,606 non-AMPs)
│       ├── MTL/                  # Multi-label dataset with 14 functional categories
│       └── Independent/             # External validation set (24,582 AMPs + 36,306 non-AMPs)
│
├── Module/
│   ├── CARE.py                      # Conservative Adaptive Representation Encoder
│   │   ├── HF-GN: High-Fidelity Group Normalization
│   │   ├── CME: Conservative Motif Extractor (kernel sizes: 3, 5)
│   │   └── ACP: Adaptive Channel Processor (SE-style attention)
│   ├── Probimamba.py                # Bidirectional selective state-space model
│   ├── ESMembedding.py              # ESM-2 embedding extractor (650M/3B/15B)
│   ├── Knowledge.py                 # 56-dim knowledge feature encoder
│   │   ├── One-hot encoding (20 amino acids)
│   │   ├── Physicochemical properties (hydrophobicity, charge, etc.)
│   │   ├── Sliding-window statistics
│   │   ├── Positional indices
│   │   └── Global features (length, isoelectric point)
│   ├── Fusion.py                    # Cross-modal attention fusion
│   └── Transformer.py               # Knowledge feature enhancement (4-layer, 8-head)
│
├── utils/
│   └── metrics.py                   # Evaluation metrics (AUROC, AUPRC, MCC, F1, etc.)
│
├── data.py                          # UnifiedProteinDataset with runtime encoding
├── model.py                         # Complete MAPLE architecture
├── train.py                         # Multi-task training script
├── evaluate.py                      # Independent evaluation & inference
├── requirements.txt                 # Python dependencies
├── outputs/                         # Training checkpoints & logs
├── eval_out/                        # Evaluation results & predictions
└── figures/                         # Visualization outputs
```

## 📊 Dataset

### Benchmark Dataset

Constructed by refining the iAMPCN dataset and integrating annotations from **9 curated AMP databases** (2021-2024):


**Dataset Composition:**

### 14 Functional Categories

1. **Antibacterial** - Activity against bacteria
2. **Antibiofilm** - Disrupts bacterial biofilms
3. **Anticancer** - Cytotoxic to cancer cells
4. **Antifungal** - Activity against fungi
5. **Anti-Gram-negative** - Targets Gram-negative bacteria
6. **Anti-Gram-positive** - Targets Gram-positive bacteria
7. **Anti-HIV** - Inhibits HIV replication
8. **Anti-MRSA** - Effective against MRSA strains
9. **Antioxidant** - Scavenges free radicals
10. **Antiparasitic** - Activity against parasites
11. **Antiviral** - Inhibits viral replication
12. **Cytotoxic** - General cellular toxicity
13. **Hemolytic** - Lyses red blood cells
14. **Anti-mammalian** - Toxic to mammalian cells

### External Validation Dataset


## 🏗 Model Architecture

### Overview

MAPLE employs a **dual-stream architecture** that processes ESM-2 embeddings and knowledge features separately before cross-modal fusion:

<div align="center">
  <img src="figures/architecture.jpg" alt="MAPLE Architecture" width="800"/>
</div>

### Core Components

#### 1. ESM-2 Embedding Extractor

- **Purpose**: Captures evolutionary and structural information from protein sequences
- **Model variants**:
  - `esm2_t48_15B_UR50D` (15B params) - Highest accuracy
  - `esm2_t36_3B_UR50D` (3B params) - Balanced performance
  - `esm2_t33_650M_UR50D` (650M params) - **Default**, good speed/accuracy tradeoff
- **Output**: Dense representations [L × 1280] encoding residue context

#### 2. Knowledge Feature Encoder

Extracts **56-dimensional** handcrafted features per residue:

```python
Features = [
    One-hot encoding (20),           # Amino acid identity
    Physicochemical (12),            # Hydrophobicity, charge, polarity, etc.
    Sliding-window stats (8),        # Local composition patterns
    Positional indices (4),          # Relative/absolute position
    Global attributes (12)           # Length, pI, molecular weight, etc.
]
```

Enhanced through a **4-layer Transformer** (8 attention heads) to capture global context.

#### 3. CARE Module (Conservative Adaptive Representation Encoder)

Three-stage local feature processing:

**a) High-Fidelity Group Normalization (HF-GN)**
```
HF-GN(X) = α · X + (1 - α) · σ(W_g) · GN(X)
```
- Stabilizes training while preserving pretrained semantics
- Learnable retention rate α controls normalization strength

**b) Conservative Motif Extractor (CME)**
```
Multi-scale extraction: Conv1D(k=3) ∥ Conv1D(k=5)
Attention weighting: Softmax(Linear(Concat(X_3, X_5)))
Conservative fusion: β · X + (1 - β) · AttentionWeighted(X_3, X_5)
```
- Parallel convolutions capture local motifs at different scales
- Attention mechanism emphasizes functional patterns
- Hyperparameter β balances new features with original input

**c) Adaptive Channel Processor (ACP)**
```
Squeeze: Global average pooling → [C]
Excite: FC → ReLU → FC → Sigmoid → [C]
Recalibrate: X · reshape(channel_weights)
```
- Dynamically adjusts feature importance per channel
- Inspired by Squeeze-and-Excitation networks

#### 4. ProBiMamba (Protein Bidirectional Mamba)

Efficient long-range dependency modeling using **Selective State-Space Models**:

**Key Advantages over RNNs:**
- **Linear complexity** O(L) vs. RNN's O(L²) attention
- **No vanishing gradients** through continuous-time formulation
- **Parallelizable** training unlike sequential RNNs
- **Input-dependent selectivity** adapts to sequence content

**Architecture:**
```
Forward SSM:  processes sequence 1→L
Backward SSM: processes sequence L→1
Bidirectional fusion: Weighted combination of both directions
```

Unlike traditional Mamba, ProBiMamba uses **pure bidirectional processing** without parallel FFN branches for cleaner feature flow.

#### 5. Cross-Modal Fusion

Aligns and integrates heterogeneous features through attention:

```
Q = Linear(ESM-features)   # Query from ESM-2 stream
K, V = Linear(Knowledge-features)  # Key, Value from knowledge stream

Attention = Softmax(QK^T / √d_k) · V
Fused = Concat(ESM-features, Attention-enhanced-knowledge)
```

This mechanism allows the model to:
- Dynamically weight knowledge features based on ESM-2 context
- Resolve semantic gaps between pretrained and handcrafted features
- Preserve modality-specific information while enabling interaction

#### 6. Prediction Head

Task-specific MLP classifiers:

```
Binary classification:     Fused → FC(512) → ReLU → Dropout(0.3) → FC(1) → Sigmoid
Multi-label prediction:    Fused → FC(512) → ReLU → Dropout(0.3) → FC(14) → Sigmoid
```

## 🚀 Quick Start

### 1. Binary AMP Classification

Identify AMPs from general peptide sequences:

```bash
python train.py \
    --data_csv data/Benchmark/AMP/AMP.csv \
    --label_cols label \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --save_dir outputs/maple_binary \
    --gpu 0
```

**Output:**
```
outputs/maple_binary/
├── best_model.pt          # Best checkpoint based on validation AUROC
├── training.log           # Detailed training metrics
├── config.json            # Hyperparameters and settings
└── metrics_history.csv    # Per-epoch performance
```

### 2. Multi-Label Functional Prediction

Predict 14 functional categories simultaneously:

```bash
python train.py \
    --data_csv data/Benchmark/MTL.csv \
    --label_cols antibacterial antibiofilm anticancer antifungal \
                 anti_Gram_negative anti_Gram_positive anti_HIV \
                 anti_MRSA antioxidant antiparasitic antiviral \
                 cytotoxic hemolytic anti_mammalian \
    --batch_size 12 \
    --epochs 100 \
    --learning_rate 5e-5 \
    --weight_decay 1e-5 \
    --patience 15 \
    --save_dir outputs/maple_multilabel \
    --gpu 0
```

**Expected Training Time:** ~8-10 hours on RTX 3090 (24GB)

### 3. Independent Evaluation

Test trained model on external validation set:

```bash
python evaluate.py \
    --model_path outputs/maple_multilabel/best_model.pt \
    --data_csv data/Benchmark/MTL.csv \
    --label_cols antibacterial antibiofilm anticancer antifungal \
                 anti_Gram_negative anti_Gram_positive anti_HIV \
                 anti_MRSA antioxidant antiparasitic antiviral \
                 cytotoxic hemolytic anti_mammalian \
    --batch_size 32 \
    --output_dir eval_out/independent_test \
    --gpu 0
```

**Output:**
```
eval_out/independent_test/
├── predictions.csv            # Per-sample predictions with confidence
├── metrics.json               # Comprehensive performance metrics
├── confusion_matrices/        # Per-class confusion matrices
│   ├── antibacterial.png
│   ├── antifungal.png
│   └── ...
└── roc_curves.png            # Multi-class ROC visualization
```

### 4. Quick Prediction on Custom Sequences

```python
from model import MAPLE
import torch

# Load trained model
model = MAPLE.load_from_checkpoint("outputs/maple_binary/best_model.pt")
model.eval()

# Predict on custom sequences
sequences = [
    "KLLKLLKKLLKLLK",           # Typical AMP
    "GIGAVLKVLTTGL",            # Another AMP
    "AAAAAAAAAAAA"              # Non-AMP (likely)
]

with torch.no_grad():
    predictions = model.predict(sequences)
    
for seq, pred in zip(sequences, predictions):
    print(f"{seq}: AMP probability = {pred:.4f}")
```

## 📊 Training

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_csv` | str | *required* | Path to training CSV file |
| `--label_cols` | list[str] | *required* | Space-separated label column names |
| `--esm_model` | str | `esm2_t33_650M_UR50D` | ESM-2 model variant |
| `--batch_size` | int | 16 | Training batch size |
| `--epochs` | int | 50 | Number of training epochs |
| `--learning_rate` | float | 1e-4 | Initial learning rate |
| `--weight_decay` | float | 1e-5 | L2 regularization coefficient |
| `--patience` | int | 10 | Early stopping patience |
| `--gradient_clip` | float | 1.0 | Maximum gradient norm |
| `--dropout` | float | 0.3 | Dropout probability in MLP head |
| `--warmup_steps` | int | 500 | Linear learning rate warmup |
| `--save_dir` | str | `outputs/maple` | Checkpoint directory |
| `--gpu` | int | 0 | GPU device ID (-1 for CPU) |
| `--num_workers` | int | 4 | DataLoader worker processes |
| `--mixed_precision` | flag | False | Enable FP16 training |
| `--seed` | int | 42 | Random seed for reproducibility |

### Data Format

Your CSV file should contain:
- **Required column**: `sequence` (amino acid sequences in one-letter code)
- **Label columns**: Binary values (0/1) for each task

**Example: Binary classification (AMP.csv)**
```csv
sequence,label
KLLKLLKKLLKLLK,1
GIGAVLKVLTTGL,1
AAAAAAAAA,0
MKTIIALSYIFCLVFA,0
```

**Example: Multi-label (MTL.csv)**
```csv
sequence,antibacterial,antifungal,anticancer,hemolytic
KLLKLLKKLLKLLK,1,1,0,0
GIGAVLKVLTTGL,1,0,1,0
FLPLIGRVLSGIL,0,1,1,1
```

### Training Tips

**For binary classification:**
```bash
python train.py \
    --data_csv data/Benchmark/AMP.csv \
    --label_cols label \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --patience 10
```

**For multi-label with class imbalance:**
```bash
python train.py \
    --data_csv data/Benchmark/MTL.csv \
    --label_cols [14 functional categories] \
    --batch_size 12 \
    --learning_rate 5e-5 \
    --patience 15 \
    --weight_decay 1e-5  # Stronger regularization for multi-task
```

**For large ESM-2 models (15B):**
```bash
python train.py \
    --esm_model esm2_t48_15B_UR50D \
    --batch_size 4 \               # Reduce batch size
    --gradient_accumulation 4 \    # Simulate larger batch
    --mixed_precision              # Use FP16 to save memory
```

### Monitoring Training

**View real-time logs:**
```bash
tail -f outputs/maple_binary/training.log
```

**Example output:**
```
Epoch 1/50 - Train Loss: 0.3421 | Val Loss: 0.2156 | Val AUROC: 0.9523 | Val MCC: 0.8234
Epoch 2/50 - Train Loss: 0.2134 | Val Loss: 0.1876 | Val AUROC: 0.9687 | Val MCC: 0.8756
...
Epoch 23/50 - Train Loss: 0.0543 | Val Loss: 0.0821 | Val AUROC: 0.9989 | Val MCC: 0.9610 ✓ New best!
Early stopping triggered at epoch 33 (no improvement for 10 epochs)
```

## 🔬 Evaluation

### Comprehensive Metrics

MAPLE computes the following metrics:

**Binary Classification:**
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Accuracy, Precision, Recall, F1-score
- MCC (Matthews Correlation Coefficient)
- Specificity, Sensitivity
- Confusion Matrix

**Multi-Label Classification:**
- Per-class metrics (AUROC, AUPRC, F1, MCC)
- Micro-averaged metrics (across all samples)
- Macro-averaged metrics (across all classes)
- Hamming Loss
- Subset Accuracy

### Custom Evaluation Script

```python
from evaluate import load_model, evaluate_model
import pandas as pd

# Load trained model
model, label_cols = load_model("outputs/maple_binary/best_model.pt")

# Evaluate on custom test set
results = evaluate_model(
    model=model,
    data_csv="my_test_data.csv",
    label_cols=label_cols,
    batch_size=32,
    output_dir="eval_out/custom_test"
)

# Access metrics
print(f"Test AUROC: {results['auroc']:.4f}")
print(f"Test AUPRC: {results['auprc']:.4f}")
print(f"Test MCC: {results['mcc']:.4f}")

# Load detailed predictions
predictions = pd.read_csv("eval_out/custom_test/predictions.csv")
print(predictions.head())
```


## 📈 Results

### Binary AMP Classification (Benchmark Dataset)

**Improvements over iAMPCN:**

### Multi-Label Functional Prediction

**Overall Performance:**

**Per-Function Performance:**


### External Validation Performance

**Independent test set:**

Demonstrates strong generalization to unseen data distributions.

### Comparison with State-of-the-Art

**Key Findings:**
1. MAPLE outperforms all baselines across all metrics
2. Particularly strong on challenging categories (antifungal, antibiofilm)
3. Balanced performance between sensitivity and specificity
4. Robust to class imbalance in multi-label settings

## 🎛 Customization

### Using Different ESM-2 Models

Edit `Module/ESMembedding.py` to change the pretrained model:

```python
class ESM2Embedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda"):
        # Choose your model:
        # model_name = "esm2_t48_15B_UR50D"     # 15B params - Best accuracy
        # model_name = "esm2_t36_3B_UR50D"      # 3B params - High performance
        # model_name = "esm2_t33_650M_UR50D"    # 650M params - Default
        # model_name = "esm2_t30_150M_UR50D"    # 150M params - Fast inference
        
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        ...
```
### Adding Custom Features

Extend the knowledge encoder in `Module/Knowledge.py`:

```python
class KnowledgeEnhancedSequenceEncoder:
    def compute_features(self, sequence):
        # Existing 56-dim features
        features = self.compute_base_features(sequence)  # [L, 56]
        
        # Add your custom features
        custom_features = self.compute_custom_features(sequence)  # [L, N]
        
        # Concatenate
        enhanced_features = torch.cat([features, custom_features], dim=-1)  # [L, 56+N]
        
        return enhanced_features
    
    def compute_custom_features(self, sequence):
        """
        Example: Add secondary structure predictions, 
        solvent accessibility, disorder scores, etc.
        """
        # Your feature extraction code
        custom_feat = extract_my_features(sequence)
        return torch.tensor(custom_feat)
```

### Modifying CARE Hyperparameters

Adjust motif extraction scales and fusion weights:

```python
# In Module/CARE.py
class CARE:
    def __init__(
        self,
        in_channels=1280,
        kernel_sizes=[3, 5, 7],      # Add more scales
        alpha=0.7,                    # Normalization retention
        beta=0.6,                     # Motif fusion weight
        gamma=0.5                     # Final gate bias
    ):
        ...
```

### Fine-Tuning for Specific AMP Types

Create domain-specific datasets:

```bash
# Train on only antibacterial AMPs
python train.py \
    --data_csv data/antibacterial_only.csv \
    --label_cols antibacterial \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --save_dir outputs/maple_antibacterial

# Train on antifungal AMPs
python train.py \
    --data_csv data/antifungal_only.csv \
    --label_cols antifungal \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --save_dir outputs/maple_antifungal
```

### Transfer Learning

Fine-tune pretrained MAPLE on new tasks:

```python
from model import MAPLE

# Load pretrained model
model = MAPLE.load_from_checkpoint("outputs/maple_binary/best_model.pt")

# Freeze early layers
for param in model.esm_embedder.parameters():
    param.requires_grad = False

# Train only final layers
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate XX.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
```bash
--batch_size 4  # or 8
```

2. **Use gradient accumulation:**
```bash
--batch_size 4 \
--gradient_accumulation_steps 4  # Effective batch size = 16
```

3. **Enable mixed precision:**
```bash
--mixed_precision
```

4. **Use smaller ESM-2 model:**
```python
# In ESMembedding.py, change to:
model_name = "esm2_t30_150M_UR50D"
```

5. **Reduce sequence length:**
```python
# In data.py, add max length filter:
max_length = 100  # Truncate long sequences
```

### CUDA Not Available

**Symptoms:**
```
AssertionError: CUDA not available
```

**Solutions:**

1. **Install CUDA-compatible PyTorch:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

2. **Verify CUDA installation:**
```bash
nvidia-smi
nvcc --version
```

3. **Run on CPU (slower):**
```bash
python train.py --gpu -1
```

### Slow Training

**Symptoms:**
- Training takes >12 hours for 50 epochs

**Solutions:**

1. **Enable DataLoader optimization:**
```bash
--num_workers 4  # or 8, depending on CPU cores
```

2. **Use mixed precision:**
```bash
--mixed_precision
```

3. **Reduce validation frequency:**
```python
# In train.py, validate every N epochs
validate_every = 5
```

4. **Use faster ESM-2 model:**
```python
model_name = "esm2_t30_150M_UR50D"  # 45x faster than 15B
```

### Poor Performance on Custom Data

**Symptoms:**
- AUROC < 0.80 on validation set
- High loss, not converging

**Diagnosis:**

1. **Check data quality:**
```python
import pandas as pd
df = pd.read_csv("your_data.csv")

# Check for issues
print(f"Missing values: {df.isnull().sum()}")
print(f"Sequence lengths: {df['sequence'].str.len().describe()}")
print(f"Label distribution: {df['label'].value_counts()}")
```

2. **Verify label format:**
```python
# Labels should be 0/1, not strings
assert df['label'].dtype in [int, float]
assert df['label'].isin([0, 1]).all()
```

3. **Adjust for class imbalance:**
```bash
--pos_weight 2.0  # If AMPs are minority class
```

4. **Try different learning rates:**
```bash
--learning_rate 1e-5  # Lower for fine-tuning
--learning_rate 5e-4  # Higher for training from scratch
```

## 📚 Citation



## 🤝 Contributing

We welcome contributions to MAPLE! Areas where you can help:

1. **Bug fixes** - Report issues or submit fixes
2. **New features** - ESM-2 variants, additional feature encoders
3. **Documentation** - Improve README, add tutorials
4. **Benchmarks** - Compare on new AMP datasets
5. **Applications** - Use MAPLE for novel AMP discovery

**Contribution Workflow:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Submit a Pull Request with detailed description

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Key Points:**
- ✅ Free for academic and commercial use
- ✅ Modification and redistribution allowed
- ✅ No warranty provided
- ⚠️ Must retain copyright notice

## 🙏 Acknowledgments

MAPLE builds upon outstanding prior work:

- **ESM-2**: Evolutionary Scale Modeling from [Meta AI](https://github.com/facebookresearch/esm)
- **Mamba**: Efficient state-space models from [state-spaces](https://github.com/state-spaces/mamba)
- **iAMPCN**: Baseline framework for AMP prediction
- **AMP Databases**: dbAMP, DRAMP, APD3, LAMP2, and others

**Datasets:**
- dbAMP 2.0: [http://csb.cse.yzu.edu.tw/dbAMP/](http://csb.cse.yzu.edu.tw/dbAMP/)
- DRAMP 3.0: [http://dramp.cpu-bioinfor.org/](http://dramp.cpu-bioinfor.org/)
- APD3: [https://aps.unmc.edu/](https://aps.unmc.edu/)
- UniProt: [https://www.uniprot.org/](https://www.uniprot.org/)

**Funding:**
This work was supported by [Your Funding Sources].

## 📧 Contact

**For Questions & Support:**
- 📧 Email: 
  - Hao Liu: [your.email@university.edu]
  - Guo Yu: guoyu@cpu.edu.cn
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/MAPLE/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/MAPLE/discussions)

**Authors:**
- **Hao Liu** (First Author) - China Pharmaceutical University
- **Yi Shi** (First Author) - China Pharmaceutical University  
- **Dechuan Zhan** (Corresponding) - Nanjing University
- **Guangji Wang** (Corresponding) - China Pharmaceutical University
- **Guo Yu** (Corresponding) - China Pharmaceutical University

---

<div align="center">

**⭐ If you find MAPLE useful, please star this repository! ⭐**

[![Star History](https://img.shields.io/github/stars/yourusername/MAPLE?style=social)](https://github.com/yourusername/MAPLE/stargazers)

*Accelerating the discovery of safe, multifunctional antimicrobial peptides through AI*


</div>


