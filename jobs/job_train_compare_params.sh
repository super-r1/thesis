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
SCRATCH_DIR="/scratch-shared/$USER/thesis_train_$SLURM_JOB_ID"
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
    "all_1e4 1e-4 16 all-linear"
    "all_5e5 5e-5 16 all-linear"
    "all_2e5 2e-5 16 all-linear"
    "attn_1e4 1e-4 8 attention"
    "attn_5e5 5e-5 8 attention"
    "attn_1e5 1e-5 8 attention"
    "attn_1e6 1e-6 8 attention"
)

# make output folder in home directory
OUTPUT_PATH="$SLURM_SUBMIT_DIR/outputs/fine_tuned_model_$SLURM_JOB_ID"
mkdir -p "$OUTPUT_PATH"

# loop through experiments
# echo "Starting Gemma Training (Fine-Tuning) at $(date)"
for exp in "${experiments[@]}"; do
    read -r name lr rank layers <<< "$exp"
    
    echo "------------------------------------------------"
    echo "RUNNING EXPERIMENT: $name"
    echo "Params: LR=$lr, Rank=$rank, Layers=$layers"
    echo "------------------------------------------------"

    # run experiment
    python -u train.py --name "$name" --lr "$lr" --rank "$rank" --layers "$layers"

    LOCAL_SAVE_DIR="$SCRATCH_DIR/outputs/$name"

    # copy results for this experiment
    echo "Saving $name from $LOCAL_SAVE_DIR/checkpoint-* to $OUTPUT_PATH/$name"
    mkdir -p "$OUTPUT_PATH/$name"
    cp -r "$SCRATCH_DIR/outputs/$name"/checkpoint-*/* "$OUTPUT_PATH/$name/"
    
    # clear scratch (to save space)
    rm -rf "$SCRATCH_DIR/outputs/$name"
    
    echo "Completed $name at $(date)"
    echo ""
done

echo "All 7 runs completed at $(date)"