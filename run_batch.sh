#!/usr/bin/env bash

# Batch script to run sigma-score-estimator training for multiple loss types
# and both model variants (omega_x and omega_x_sigma).

set -e

# Number of epochs for all runs
EPOCHS=30

# List of loss types to iterate over (must match config.py valid_loss_types)
LOSS_TYPES=(
  "omega_hat"
  "omega_epsilon"
  "omega_chi_approx"
  "omega_chi_mean"
  "sigma_direct"
  "sigma_normalized"
  "sigma_relative"
  "sigma_calibrated"
)

# List of model types to iterate over
MODEL_TYPES=(
  "omega_x"
  "omega_x_sigma"
)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

for MODEL in "${MODEL_TYPES[@]}"; do
  for LOSS in "${LOSS_TYPES[@]}"; do
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    EXP_DIR="experiments/train/${MODEL}/${LOSS}_${TIMESTAMP}"

    echo "============================================================"
    echo "Running training: model=${MODEL}, loss_type=${LOSS}, epochs=${EPOCHS}"
    echo "Saving results to: ${EXP_DIR}"
    echo "============================================================"

    python main.py train \
      --model-type "${MODEL}" \
      --loss-type "${LOSS}" \
      --epochs "${EPOCHS}" \
      --exp-dir "${EXP_DIR}"

    echo
  done
done

echo "All batch runs completed."


