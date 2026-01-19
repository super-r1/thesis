# Machine Translation Thesis

## About
This repository contains the code for my thesis project on Machine Translation.

## Code Structure
```text
├── main.py              # Run full pipeline
├── src/                 # Source code folder
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration (Model IDs, Devices)
│   ├── data_loader.py   # Dataset loading and processing
│   ├── model_utils.py   # Translation Model loading
│   ├── translate.py     # Run translations
│   └── evaluate.py      # Metric calculations
├── notebooks/           # Experimental and exploratory Jupyter Notebooks
├── outputs/             # (Created at runtime) Saved translations and scores
└── .env                 # Environment variables (KEEP PRIVATE)
```

## Usage

### 1. Environment Setup
`pip install -r requirements.txt`

### 2. Configuration
Add your HuggingFace `HF_TOKEN` in the `.env` file.

### 3. Running the Pipeline
`python main.py`