#!/bin/bash
#SBATCH --job-name=gemma_pipeline
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j_pipeline.out

# load environment
module purge
module load 2024
module load CUDA/12.6.0

export PYTHONWARNINGS="ignore:pkg_resources is deprecated"

# set up scratch storage for faster performance
SCRATCH_DIR="/scratch-shared/$USER/pipeline_$SLURM_JOB_ID"
export HF_HOME="/scratch-shared/$USER/hf_cache"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HF_HOME" 

# sync code to scratch
rsync -a --exclude='.git' --exclude='outputs' --exclude='logs' . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

source activate translate-gemma
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# training part
DATA_AGAIN="/scratch-shared/bveenman/data/translate_again/flores_CLEAN"
echo "--- Starting Fine-Tuning ---"

# use log file for output to capture checkpoint path
TRAIN_LOG="train_output.log"

# run fine-tuning (with output going to train_log file)
python -u train.py \
    --name "run_$SLURM_JOB_ID" \
    --langs nl zh \
    --mode again \
    --data_folder "$DATA_AGAIN" 2>&1 | tee "$TRAIN_LOG"

# grab checkpoint path from train.py output
# looks for the most recent subfolder in the output directory that contains "checkpoint-" in its name
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

# save results
echo "Moving results to Home directory..."
mkdir -p "$SLURM_SUBMIT_DIR/outputs/pipeline_run_$SLURM_JOB_ID"
cp -r outputs/* "$SLURM_SUBMIT_DIR/outputs/pipeline_run_$SLURM_JOB_ID/"

echo "Job completed at $(date)"