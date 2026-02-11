#!/bin/bash
#SBATCH --job-name=pipeline_compare
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_pipe_compare.out

# load environment
module purge
module load 2024
module load CUDA/12.6.0

export PYTHONWARNINGS="ignore:pkg_resources is deprecated"

# set up scratch storage for faster performance
SCRATCH_DIR="/scratch-shared/$USER/compare_$SLURM_JOB_ID"
export HF_HOME="/scratch-shared/$USER/hf_cache"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HF_HOME" 

# sync code to scratch
rsync -a --exclude='.git' --exclude='outputs' --exclude='logs' . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# activate conda environment
source activate translate-gemma
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# create experiment configurations
# "exp_name learning_rate rank layers"
experiments=(
    "2e-5 2e-5 16 all-linear"
    "1e-5 1e-5 16 all-linear"
    "5e-6 5e-6 16 all-linear"
    "2e-6 2e-6 16 all-linear"
    "1e-6 1e-6 16 all-linear"
)

# set data path for translate again data
DATA_AGAIN="/scratch-shared/bveenman/data/translate_again/flores_CLEAN"

# loop through experiments
for exp in "${experiments[@]}"; do
    read -r name lr rank layers <<< "$exp"

    echo "------------------------------------------------"
    echo "RUNNING EXPERIMENT: $name"
    echo "Params: LR=$lr, Rank=$rank, Layers=$layers"
    echo "------------------------------------------------"

    # training part
    echo "--- Starting Fine-Tuning ---"
    TRAIN_LOG="log_train_${name}.log"

    # run fine-tuning (with experiment specific params)
    python -u train.py \
        --name "$name" \
        --lr "$lr" \
        --rank "$rank" \
        --layers "$layers" \
        --langs nl zh \
        --mode again \
        --data_folder "$DATA_AGAIN" 2>&1 | tee "$TRAIN_LOG"

    # grab checkpoint path from train.py output
    BASE_DIR=$(grep "COMPLETED_CHECKPOINT:" "$TRAIN_LOG" | cut -d':' -f2 | xargs)
    CHECKPOINT_PATH=$(ls -td "${BASE_DIR}/checkpoint-"* 2>/dev/null | head -1)

    # build the argument string: only add the flag if the path exists
    if [ -n "$CHECKPOINT_PATH" ]; then
        echo "Detected Checkpoint: $CHECKPOINT_PATH"
        CHECKPOINT_ARG="--checkpoint $CHECKPOINT_PATH"
    else
        echo "No checkpoint folder detected. Python will load the base model (None)."
        CHECKPOINT_ARG=""
    fi

    # inference & evaluation part
    echo "--- Starting Inference & Evaluation ---"
    python -u main.py \
        $CHECKPOINT_ARG \
        --langs nl zh \
        --num_samples 5 \
        --mode again \
        --force 

    # save results for this specific experiment
    echo "Moving $name results to Home directory..."
    FINAL_DEST="$SLURM_SUBMIT_DIR/outputs/compare_$SLURM_JOB_ID/$name"
    mkdir -p "$FINAL_DEST"
    cp -r outputs/* "$FINAL_DEST/"
    cp "$TRAIN_LOG" "$FINAL_DEST/"

    # clear scratch outputs (strictly necessary to avoid cross-run cache issues)
    rm -rf outputs/*
    
    echo "Completed $name at $(date)"
    echo ""
done

echo "Comparison run completed at $(date)"