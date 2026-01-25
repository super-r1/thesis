#!/bin/bash
#SBATCH --job-name=translate_gemma
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out

# load environment
module purge
module load 2024
module load CUDA/12.6.0

export PYTHONWARNINGS="ignore:pkg_resources is deprecated"

# set up scratch storage for faster performance
SCRATCH_DIR="/scratch-shared/$USER/thesis_run_$SLURM_JOB_ID"
export HF_HOME="/scratch-shared/$USER/hf_cache"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HF_HOME" 
cp -r . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# activate conda environment
source activate translate-gemma
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# create 7 different experiment configurations
# "exp_name learning_rate rank layers"
experiments=(
    "all_1e4"
    "all_5e5"
    "all_2e5"
    "attn_1e4"
    "attn_5e5"
    "attn_1e5"
    "attn_1e6"
)

# get parent folder for checkpoints
CHECKPOINT_BASE="$SLURM_SUBMIT_DIR/outputs/compare_lr_train"

# run through checkpoints
echo "Starting Gemma Translation at $(date)"

for name in "${experiments[@]}"; do
    echo "------------------------------------------------"
    echo "EVALUATING: $name"
    echo "------------------------------------------------"

    # run experiment
    python main.py --checkpoint "$CHECKPOINT_BASE/$name"

    # save results
    echo "Moving results to Home directory..."
    TIMESTAMP=$(date +"%Y%m%d_%H%M")
    cp -r outputs/* "$SLURM_SUBMIT_DIR/outputs/"
    
    # clear local outputs to prevent mixing results
    rm -rf outputs/*
done

echo "Job completed at $(date)"