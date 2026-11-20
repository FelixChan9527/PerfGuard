#!/usr/bin/env bash
AGENT4AIGC_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT4AIGC_CONDA_ROOT="${AGENT4AIGC_CONDA_ROOT:-/data/chenzhipeng/miniconda3}"
source "$AGENT4AIGC_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate Agent4AIGC
export PYTHONNOUSERSITE=1
export AGENT4AIGC_MODEL_ROOT="${AGENT4AIGC_MODEL_ROOT:-$AGENT4AIGC_PROJECT_ROOT/external/Step1X-Edit/models}"
export AGENT4AIGC_CONDA_ENVS_ROOT="${AGENT4AIGC_CONDA_ENVS_ROOT:-$AGENT4AIGC_CONDA_ROOT/envs}"
export AGENT4AIGC_DEVICE="${AGENT4AIGC_DEVICE:-auto}"
if [[ "$AGENT4AIGC_DEVICE" == npu* || "$AGENT4AIGC_DEVICE" == auto ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
