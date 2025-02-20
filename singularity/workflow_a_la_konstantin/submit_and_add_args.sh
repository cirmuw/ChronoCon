#!/bin/bash

# Check if at least one argument (the job script) is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 slurm_script [args...]"
    exit 1
fi

SLURM_SCRIPT="$1"
shift  # Shift the arguments to exclude the job script name
ARGS=("$@")  # All remaining arguments

# Extract the base name of the job script without extension
JOB_SCRIPT_BASENAME=$(basename "$SLURM_SCRIPT")
JOB_SCRIPT_NAME="${JOB_SCRIPT_BASENAME%.*}"

# Process arguments: replace '/' with '-' in each argument
PROCESSED_ARGS=()
for arg in "${ARGS[@]}"; do
    arg="${arg//\//-}"       # Replace '/' with '-'
    PROCESSED_ARGS+=("$arg")
done

# Join the processed arguments with '--'
ARGS_JOINED=$(IFS='--' ; echo "${PROCESSED_ARGS[*]}")

# Create the job name
JOB_NAME="${JOB_SCRIPT_NAME}__${ARGS_JOINED}"

# Remove any invalid characters for a job name
JOB_NAME=${JOB_NAME//[^A-Za-z0-9_-]/}

echo "Submitting job with name: $JOB_NAME"

# Submit the job with all arguments
sbatch --job-name="$JOB_NAME" "$SLURM_SCRIPT" "${ARGS[@]}"

