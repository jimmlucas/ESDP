## ESDP Installation

### Prerequisites

- **Python**: 3.8 or higher  
- **Operating System**: Linux, macOS  
- **RAM**: Minimum 8GB recommended  
- **Storage**: ~3GB for dependencies and models  

---

### Install with pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing.git
cd ESDP-Early-Stop-Decision-Polishing

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows (WSL): source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### Setup with conda

```bash
# Clone the repository
git clone https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing.git
cd ESDP-Early-Stop-Decision-Polishing

# Create conda environment
conda env create -f environment.yml
conda activate ont-polishing

# Verify installation
python -c "import xgboost, sklearn, pandas; print('✓ Installation successful')"
```

### Dependencies

<details>
<summary>Click to see full dependency list</summary>

```
# Core ML Libraries
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0

# Ordinal Classification
mord>=0.6

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Configuration & Utilities
pyyaml>=5.4.0
joblib>=1.0.0

# Optional (for extended features)
shap>=0.40.0      # For model interpretability
optuna>=2.10.0    # For hyperparameter tuning
```
