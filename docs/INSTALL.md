## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux or macOS
- **RAM**: 8 GB recommended
- **Storage**: additional space required for dependencies, model artifacts, and benchmark outputs

---

### Install with pip

```bash
# Clone the repository
git clone https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing.git
cd ESDP-Early-Stop-Decision-Polishing

# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, pandas, joblib; print('Installation successful')"

```

### Setup with conda

```bash

# Clone the repository
git clone https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing.git
cd ESDP-Early-Stop-Decision-Polishing

# Create the conda environment
conda env create -f environment.yml
conda activate ont-polishing

# Verify installation
python -c "import sklearn, pandas, joblib; print('Installation successful')"

```

### Dependencies

<details>
<summary>Click to see full dependency list</summary>

```
# Core libraries
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
pyyaml>=5.4.0

# Model development and evaluation
xgboost>=1.5.0
imbalanced-learn>=0.9.0
mord>=0.6

# Visualization and analysis
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Optional
shap>=0.40.0
optuna>=2.10.0

```
</details>