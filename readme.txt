# MHGCL: Multimodal heterogeneous graph contrastive learning for cancer patient classification

This repository provides the official implementation of MHGCL, a multimodal heterogeneous graph contrastive learning framework tailored for dynamic feature selection and cross-modal alignment tasks on biomedical multimodal data.

---

**MHGCL addresses this by:**
- designing dynamic graph learning for selecting high revelant features
- performing contrastive learning on multimodal heterogeneous GNN backbones
- using cross-attention fusion to fusion multi-modal representations
---

## 📁 Repository Structure
MHGCL/
├── data/
│ └── BRCA/ # BRCA multi-omics dataset and preprocessing files
├── CrossAttentionFusion.py # Cross-modal attention-based feature fusion module
├── DynamicalGraphLearning.py # Dynamic patient graph construction and learning
├── MHGCL.py # Core implementation of the MHGCL model
├── ModelEvaluate.py # Evaluation metrics and pipeline (classification/regression)
├── MultiViewHeteroGNNContrastiveLearning.py # Multi-view heterogeneous GNN backbone
├── base_layer.py # Basic neural network building blocks
├── utils.py # Helper functions (seed control, logging, metrics, etc.)
├── train.py # Main training script
└── README.md # This document


---

## ⚙️ Environment & Dependencies
The code is developed and tested with **Python 3.9.23**.  
We recommend using a virtual environment (e.g., conda) to manage dependencies:

```bash
# Create and activate a new conda environment
conda create -n mhgcl python=3.9.23
conda activate mhgcl

# Install required packages
pip install torch==2.5.1
pip install torch-geometric==2.6.1
pip install pandas==2.3.1
pip install numpy==1.26.4
pip install scikit-learn==1.6.2


📥 Data Preparation
This repository uses the BRCA multi-omics dataset as an example.


# Usage
We provide the scripts for running MHGCL
```
python train.py
```

# Seed
We set global random seed to 2025
