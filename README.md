# Hybrid Quantum-Classical Framework for Multi-Label & Single-Label Bengali Emotion Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-QML-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation, experimental notebooks, dataset preparation scripts, time complexity benchmarks, and paper source files for the research work:

> **"Hybrid Quantum-Classical Framework for Multi-Label Bengali Emotion Classification"**  
> *Tathagata Mitra, Biswanath Ranjit, Pritam Pal, and Dipankar Das*  
> Departments of Physics and Computer Science, Jadavpur University, Kolkata, India.

---

## Overview

Multi-label emotion classification for low-resource and code-mixed languages such as Bengali poses significant challenges due to limited annotated corpora, severe class imbalance, and complex co-occurring emotions. 

This project investigates whether inserting **Parameterized Quantum Circuits (PQCs)** / **Variational Quantum Circuits (VQCs)** as trainable bottleneck representations inside classical transformer architectures can enhance feature expressivity and multi-label classification accuracy. 

By coupling pretrained Bengali transformer encoders (**IndicBERTv2** and **BanglaBERT**) with variational quantum circuits across **2-qubit, 4-qubit, and 8-qubit registers**, this framework evaluates the interaction between quantum feature encoding, entanglement topology, circuit depth, and classical representation quality.

---

## Repository Structure

The codebase is organized into four main functional directories:

```
VQC-CODES-main/
├── MULTI_LABEL/               # Multi-label emotion classification pipeline (EmoNoBa)
│   ├── BASE NOTEBOOKS/        # Core Jupyter notebooks across circuit designs
│   │   ├── CUSTOM_1/          # Custom 1 (Cross-coupling entangling VQC) notebooks (2Q, 4Q)
│   │   ├── CUSTOM_2/          # Custom 2 (Data re-uploading rotating VQC) notebooks (2Q, 4Q, 8Q)
│   │   └── SEL/               # Strongly Entangling Layers (SEL) baseline notebooks & circuits
│   └── DATASET/               # Multi-label dataset files & extended evaluation metrics script
│       ├── EmoNoBa_Dataset/   # Train, validation, and test splits (CSV format)
│       └── evaluation_schemes.py # Multi-metric evaluation protocol (EMR, Partial Match, F1)
│
├── SINGLE_LABEL/              # Single-label emotion classification pipeline (BanglaEmotion)
│   ├── BASE NOTEBOOKS/        # Single-label experimental notebooks
│   │   ├── CUSTOM_1/          # Custom 1 VQC notebooks with BanglaBERT & IndicBERT
│   │   ├── CUSTOM_2/          # Custom 2 VQC notebooks (8-qubit register)
│   │   └── SEL/               # SEL 4-qubit baseline notebooks
│   └── DATASET/               # BanglaEmotion raw textual datasets (train, test, corpus)
│
├── TIME COMPLEXITY/           # Execution time & computational complexity profiling
│   ├── CUSTOM_1/              # 4-qubit Custom 1 execution time benchmarks
│   ├── CUSTOM_2/              # 8-qubit Custom 2 execution time benchmarks
│   └── SEL/                   # 4-qubit SEL execution time benchmarks
│
├── MISCELLANEOUS/             # Research paper LaTeX source files, PDFs, and references
│   ├── MY-DRAFT-LATEX.tex     # Full paper LaTeX source file
│   ├── MY-DRAFT-LATEX.pdf     # Compiled PDF of the paper manuscript
│   ├── refs.bib               # Complete BibTeX bibliography file
│   └── REFERENCE_PAPERS/      # Curated reference literature on QML & NLP
│
└── .gitignore                 # Exclusion rules for temporary build & tool files
```

---

## Quantum Architectures & Circuit Designs

The framework evaluates three distinct variational quantum circuit (VQC) architectures:

### 1. Strongly Entangling Layers (SEL) Baseline
- **Encoding**: Angle encoding mapping classical features into single-qubit rotations.
- **Ansatz**: Multi-layer parametric rotation gates ($R_Z(\theta_1) R_Y(\theta_2) R_Z(\theta_3)$) coupled with nearest-neighbor CNOT ring entanglers.
- **Configurations**: Evaluated on 2-qubit, 4-qubit, and 8-qubit registers.

### 2. Custom 1 Architecture (Biswanath Scheme)
- **Topology**: Cross-coupling entangling scheme featuring all-to-all CNOT gate pairings.
- **Encoding**: Feature re-uploading mechanism interleaved with trainable unitary rotations.
- **Focus**: Maximizing entanglement entropy and inter-qubit correlation across multi-label feature maps.

### 3. Custom 2 Architecture (Tathagata Scheme)
- **Topology**: Data re-uploading encoding scheme with a rotating ansatz.
- **Encoding**: Interleaved feature re-uploading layers $U(x)$ and trainable parameter unitaries $U(\theta)$ to boost expressivity without increasing qubit counts.
- **Focus**: High expressivity and resilience against barren plateaus, achieving optimal multi-label classification accuracy on 8-qubit registers.

---

## Encoders & Datasets

### Classical Encoders
- **IndicBERTv2** (`ai4bharat/indic-bert-v2-mBERT`): Multilingual transformer model optimized for Indic languages.
- **BanglaBERT** (`csebuetnlp/banglabert`): Pretrained BERT model fine-tuned specifically for Bengali NLP tasks.

### Datasets
- **EmoNoBa**: A 6-way multi-label emotion dataset for Bengali text, capturing complex co-occurring emotions (e.g., joy, sadness, anger, fear, surprise, disgust).
- **BanglaEmotion**: A single-label benchmark dataset used to evaluate model generalizability across single-label emotion targets.

---

## Evaluation Protocol & Metrics

Evaluating multi-label classification requires measures beyond standard accuracy. This repository implements a unified metric suite via [`evaluation_schemes.py`](MULTI_LABEL/DATASET/evaluation_schemes.py):

1. **Standard Metrics (Scheme 1)**:
   - Per-label Macro, Micro, and Weighted $F_1$-scores.
   - Hamming Loss (fraction of incorrectly predicted label pairs).

2. **Exact Match Ratio (EMR) (Scheme 2)**:
   - Strict metric where a sample scores 1 only if all 6 label predictions perfectly match the ground truth.
   - Precision = Recall = $F_1$ = **EMR** ($P = R = F_1 = \text{EMR}$).

3. **Partial Match / Instance-Level (Scheme 3)**:
   - Computes set-based overlap (Precision, Recall, $F_1$) per text instance and averages across all samples to reward partial label set overlap.

---

## Key Empirical Results

- **Baselines vs. Quantum Enhancements**: Inserting the VQC bottleneck significantly enhances performance on weaker baseline representations.
- **BanglaBERT Uplift**: Adding Custom 2 VQC boosts BanglaBERT's Macro $F_1$-score from **0.2101 to 0.4850** and raises the Exact Match Ratio (EMR) from **0.0000 to 0.4630**.
- **Top Performing Setup**: **IndicBERTv2 + Custom 2 VQC at 8 qubits** achieves the overall best performance:
  - **Macro $F_1$-Score**: `0.5165`
  - **Exact Match Ratio (EMR)**: `0.4577`

---

## Prerequisites & Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- PennyLane 0.30+
- HuggingFace Transformers
- Scikit-Learn
- NumPy, Pandas, Matplotlib

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/tmdeveloper007/VQC-CODES.git
cd VQC-CODES

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install torch pennylane transformers scikit-learn pandas numpy matplotlib
```

---

## Running the Code

### Running Multi-Label Training & Notebooks
Open any notebook in JupyterLab or VSCode:
```bash
jupyter lab MULTI_LABEL/BASE\ NOTEBOOKS/CUSTOM_2/4-qubit-banglabert1.ipynb
```

### Running Extended Evaluation Standalone
```python
from MULTI_LABEL.DATASET.evaluation_schemes import exact_match_scores, partial_match_scores

# gold: (N, 6) binary array, preds: (N, 6) binary array
emr_results = exact_match_scores(gold_labels, predictions)
partial_results = partial_match_scores(gold_labels, predictions)

print(f"Exact Match Ratio (EMR): {emr_results['exact_match_ratio']:.4f}")
print(f"Partial Match F1: {partial_results['f1']:.4f}")
```

### Compiling Research Paper LaTeX
```bash
cd MISCELLANEOUS
pdflatex MY-DRAFT-LATEX.tex
bibtex MY-DRAFT-LATEX
pdflatex MY-DRAFT-LATEX.tex
pdflatex MY-DRAFT-LATEX.tex
```

---

## Citation

If you find this codebase or research useful in your work, please cite:

```bibtex
@article{mitra2026hybrid,
  title={Hybrid Quantum-Classical Framework for Multi-Label Bengali Emotion Classification},
  author={Mitra, Tathagata and Ranjit, Biswanath and Pal, Pritam and Das, Dipankar},
  journal={Department of Physics and Department of Computer Science, Jadavpur University},
  year={2026}
}
```

---

## License

This repository is distributed under the [MIT License](LICENSE).
