# Machine Translation Thesis

## About
This repository contains the code for my thesis project on Machine Translation.

## Code Structure
```text
├── main.py               # Run translation and evaluation pipeline
├── train.py              # Run fine-tuning pipeline
├── src/                  # Source code folder
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration (Model IDs, Devices)
│   ├── data_loader.py    # Dataset loading and processing
│   ├── model_utils.py    # Translation Model loading
│   ├── translate.py      # Run translations
│   ├── metricx_models.py # Models for MetricX
│   └── evaluate.py       # Metric calculations
├── jobs/                 # Bash job scripts for Snellius Cluster
├── outputs/              # (Created at runtime, ignored by Git) Saved results
├── envs/                 # Environment setup files
└── .env                  # Environment variables (KEEP PRIVATE)
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

Set `DATA_DIR` and `OUTPUT_DIR` in `src/config.py`

### 3. Running the Pipeline
##### Inference and evaluation
Locally: `python main.py` (use `-h` to see options)

On cluster: `sbatch job_main.sh`

##### Fine-Tuning
Locally: `python train.py` (use `-h` to see options)

On cluster: `sbatch job_train.sh`