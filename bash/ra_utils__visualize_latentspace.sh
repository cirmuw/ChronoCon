#!/usr/bin/env bash
# --------------------------------------------------------------------
# Wrapper for ra_utils__visualize_latentspace
# Usage: ./run_visualize_latentspace.sh [all args...]
# Example: ./run_visualize_latentspace.sh --config cfg.yml -h


# Add to .bashrc as 
# alias ra_utils__visualize_latentspace_fs='/home/cwatzenboeck/code/RA/ra_utils/bash/ra_utils__visualize_latentspace.sh'
# --------------------------------------------------------------------

# Exit on error
set -e

# Export required environment variable
export CSEG_UTILS_TORCH_MP_SHARING_STRATEGY="file_system"

# Forward all arguments to the underlying command
ra_utils__visualize_latentspace "$@"


