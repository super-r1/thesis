#!/bin/bash
#SBATCH --job-name=train_gemma
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_train.out

# load environment
module purge
module load 2024
module load CUDA/12.6.0

export PYTHONWARNINGS="ignore:pkg_resources is deprecated"

# set up scratch storage for faster performance
SCRATCH_DIR="/scratch-shared/$USER/$SLURM_JOB_ID"
export HF_HOME="/scratch-shared/$USER/hf_cache"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HF_HOME" 
#cp -r . "$SCRATCH_DIR"
rsync -av --exclude='outputs' --exclude='logs' . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# activate conda environment
source activate translate-gemma
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# set data path for translate again data
DATA_AGAIN="/scratch-shared/bveenman/data/translate_again/flores"

# run training
echo "Starting Gemma Training (Fine-Tuning) at $(date)"
python -u train.py --name mult_samples_1_again --langs nl zh --mode again --data_folder $DATA_AGAIN --checkpoint "$SLURM_SUBMIT_DIR/outputs/mult_samples_1/checkpoint-1006"

# get OUTPUT_DIR from config
OUTPUT_DIR=$(python -c "from src.config import OUTPUT_DIR; print(OUTPUT_DIR)")

# make output folder in home directory
OUTPUT_PATH="$SLURM_SUBMIT_DIR/outputs/fine_tuned_model_$SLURM_JOB_ID"
mkdir -p "$OUTPUT_PATH"

# copy results
echo "Moving trained model from $OUTPUT_DIR to $OUTPUT_PATH"
cp -r "$OUTPUT_DIR"/* "$OUTPUT_PATH/"

echo "Job completed at $(date)"