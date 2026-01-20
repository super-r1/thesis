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
├── outputs/             # (Created at runtime, ignored by Git) Saved results
└── .env                 # Environment variables (KEEP PRIVATE)
```

## Usage

### 1. Environment Setup
If GPU available:
`conda env create -f environment-gpu.yml`

If no GPU available:
`conda env create -f environment.yml`

`conda activate translate-gemma`

### 2. Configuration
Add your HuggingFace `HF_TOKEN` in the `.env` file.

### 3. Running the Pipeline
Locally: `python main.py`

On cluster: `sbatch job.sh`