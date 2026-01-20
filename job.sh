#!/bin/bash
#SBATCH --job-name=translate_gemma
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

# load environment
module purge
module load 2024
module load CUDA/12.4.1  

# set up scratch storage for faster performance
SCRATCH_DIR="/scratch-shared/$USER/thesis_run_$SLURM_JOB_ID"
export HF_HOME="/scratch-shared/$USER/hf_cache"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HF_HOME" 
cp -r . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# activate conda environment
source activate translate-gemma

# run code
echo "Starting Gemma Translation at $(date)"
python main.py --limit 10

# save results
echo "Moving results to Home directory..."
TIMESTAMP=$(date +"%Y%m%d_%H%M")
cp -r outputs/* "$SLURM_SUBMIT_DIR/outputs/"

echo "Job completed at $(date)"