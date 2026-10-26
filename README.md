# AIML_PROJECT

A production-ready, modular architecture for Artificial Intelligence & Machine Learning projects following MLOps industry best practices.

---

## 📁 Directory Structure

```text
AIML_PROJECT/
│
├── config/                      # Configuration files
│   ├── config.yaml              # Paths, dataset locations, and environment configs
│   └── model_params.yaml        # Model hyperparameters & training configurations
│
├── data/                        # Project datasets (managed & git-ignored)
│   ├── raw/                     # Original, immutable raw data
│   ├── interim/                 # Cleaned / intermediate transformed data
│   ├── processed/               # Final training & testing datasets
│   └── external/                # External or third-party datasets
│
├── docs/                        # Project documentation & design docs
│
├── models/                      # Trained model binaries, checkpoints, and scalers
│
├── notebooks/                   # Jupyter notebooks for EDA and rapid prototyping
│   └── 01_exploratory_data_analysis.ipynb
│
├── reports/                     # Generated analysis reports & evaluation figures
│   └── figures/                 # Visualizations, ROC curves, confusion matrices
│
├── src/                         # Core Python source code
│   ├── __init__.py
│   ├── components/              # Modular pipeline stages
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Reads raw data and splits train/test sets
│   │   ├── data_transformation.py # Feature engineering, scaling, pipelines
│   │   ├── model_trainer.py     # Model training & hyperparameter tuning
│   │   └── model_evaluation.py  # Model evaluation & metrics reporting
│   ├── entity/                  # Strongly-typed configuration & artifact dataclasses
│   │   ├── __init__.py
│   │   └── config_entity.py
│   ├── pipeline/                # End-to-end orchestration workflows
│   │   ├── __init__.py
│   │   ├── training_pipeline.py # Training workflow coordinator
│   │   └── prediction_pipeline.py # Inference & scoring service
│   └── utils/                   # Shared utility modules
│       ├── __init__.py
│       ├── common.py            # YAML loader, object serializing, helpers
│       ├── exception.py         # Custom exception handling with traceback details
│       └── logger.py            # Rotating file and console logger
│
├── tests/                       # Automated unit and integration tests
│   ├── __init__.py
│   └── test_data.py
│
├── .env.example                 # Environment variables template (API keys, secrets)
├── .gitignore                   # Ignore dataset files, models, virtualenvs, cache
├── requirements.txt             # Project Python dependencies
├── setup.py                     # Setup script to install 'src' as an editable package
└── README.md                    # Project overview & usage instructions
```

---

## 🚀 Getting Started

### 1. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
*(Installing with `-e .` links `src` as an editable package, allowing you to run `from src.utils.logger import logging` from anywhere in the project.)*

### 3. Environment Variables
Copy `.env.example` to `.env` and fill in any required tokens:
```bash
copy .env.example .env
```

### 4. Running the Pipelines

#### Run the End-to-End Training Pipeline:
```bash
python -m src.pipeline.training_pipeline
```

#### Run Tests:
```bash
pytest tests/
```

---

## 🛠️ Key Design Principles

1. **Separation of Concerns**: Preprocessing, model training, and prediction are decoupled into separate components.
2. **Config-Driven**: Paths and hyperparameters live in `config/*.yaml` files rather than hardcoded in Python code.
3. **Data Versioning & Safety**: Raw data remains untouched; processed sets are stored separately and ignored by Git.
4. **Reproducibility**: Seed values, hyperparameter logging, and centralized logging ensure deterministic, debuggable runs.
