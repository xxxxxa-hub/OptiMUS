#!/bin/bash
#SBATCH --job-name=optimus_multi
#SBATCH --output=logs/optimus_%A_%a.out
#SBATCH --error=logs/optimus_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================
# CONFIGURATION - Modify these arrays as needed
# ============================================

# Datasets to run (paths relative to OptiMUS directory)
DATASETS=(
    "dataset/ComplexLP"
    "dataset/IndustryOR"
    "dataset/BWOR"
    # Add more datasets here
)

# Models to test
MODELS=(
    # "o4-mini"
    "gpt-4.1"
    
    # Add more models here
)

# Other parameters
NUM_WORKERS=50
DEVMODE=1
ERROR_CORRECTION=1
RAG_MODE=""  # Leave empty for no RAG, or set to a valid RAG mode

# ============================================
# EXECUTION
# ============================================

cd /hpc/group/fanglab/xx102/OptiMUS

echo "=========================================="
echo "OptiMUS Multi-Dataset Multi-Model Runner"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "=========================================="

# Track results
RESULTS_FILE="experiment_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Experiment Results - $(date)" > "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Running: Dataset=$DATASET, Model=$MODEL"
        echo "=========================================="

        # Build the command
        CMD="python main.py --all-dirs --data-path $DATASET --model $MODEL --num-workers $NUM_WORKERS --devmode $DEVMODE --error-correction $ERROR_CORRECTION"

        # Add RAG mode if specified
        if [ -n "$RAG_MODE" ]; then
            CMD="$CMD --rag-mode $RAG_MODE"
        fi

        echo "Command: $CMD"
        echo "Started at: $(date)"

        START_TIME=$(date +%s)
        eval "$CMD"
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Log result
        if [ $EXIT_CODE -eq 0 ]; then
            STATUS="SUCCESS"
        else
            STATUS="FAILED (exit code: $EXIT_CODE)"
        fi

        echo "Finished at: $(date) - Duration: ${DURATION}s - Status: $STATUS"
        echo "Dataset=$DATASET, Model=$MODEL, Duration=${DURATION}s, Status=$STATUS" >> "$RESULTS_FILE"
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
