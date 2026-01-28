#!/bin/bash
#SBATCH --job-name=train_gemma
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
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
cp -r . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# activate conda environment
source activate translate-gemma
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# run training
echo "Starting Gemma Training (Fine-Tuning) at $(date)"
python -u train.py --name all_1e6 --lr 1e-6 --rank 16 --layers all-linear

# get OUTPUT_DIR from config
OUTPUT_DIR=$(python -c "from src.config import OUTPUT_DIR; print(OUTPUT_DIR)")

# make output folder in home directory
OUTPUT_PATH="$SLURM_SUBMIT_DIR/outputs/fine_tuned_model_$SLURM_JOB_ID"
mkdir -p "$OUTPUT_PATH"

# copy results
echo "Moving trained model from $OUTPUT_DIR to $OUTPUT_PATH"
cp -r "$OUTPUT_DIR"/* "$OUTPUT_PATH/"

echo "Job completed at $(date)"