#!/bin/bash

# Define base paths
CKPT_A="/mnt/Home2/projects/segsynthtemplate/None/logs/fetaall_synth_ftreal/runs/2025-05-20_18-44-49/checkpoints/epoch_1322_val_dsc_0.8658_step_138915.ckpt"
CKPT_B="/mnt/Home2/projects/segsynthtemplate/None/logs/fetaall_synth/runs/2025-05-09_18-19-58/checkpoints/epoch_949_val_dsc_0.8292_step_99750.ckpt"
SCRIPT_PATH="segsynthtemplate/utils/interpolate_models.py"
OUTPUT_ROOT="/mnt/Home2/projects/segsynthtemplate/None/logs"

# Loop over interpolation weights
for alpha in $(seq 0 0.1 1.0); do
  # Format alpha to avoid issues (e.g., 0.1 -> 01)
  alpha_str=$(printf "%.1f" "$alpha")
  alpha_nodot=$(echo "$alpha_str" | sed 's/0\.//')

  # Construct output directory and filename
  OUT_DIR="${OUTPUT_ROOT}/fetaall_synth_ftreal_interp${alpha_nodot}/runs/2025-05-20_18-44-49/checkpoints"
  OUT_FILE="${OUT_DIR}/interp${alpha_nodot}.ckpt"

  # Create directory if it does not exist
  mkdir -p "$OUT_DIR"

  # Run interpolation
  python "$SCRIPT_PATH" "$CKPT_A" "$CKPT_B" "$OUT_FILE" --alpha "$alpha"
done
