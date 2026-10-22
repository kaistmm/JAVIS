#!/usr/bin/env bash
# Merge a LoRA + motion_cfm_head checkpoint into a standalone export (CPU by default; the GPU path once died at save).
#   CONFIG_FILE=configs/merge_stage1.yaml bash training/run_merge.sh
#   CONFIG_FILE=configs/merge_stage2.yaml bash training/run_merge.sh
set -e
export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"   # no colon: unset stays CPU-only
export PYTHONPATH="${PROJECT_ROOT}/training/plugin:${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONBREAKPOINT=0

CONFIG_FILE="${CONFIG_FILE:?set CONFIG_FILE to configs/merge_stage1.yaml or configs/merge_stage2.yaml}"
cd "${PROJECT_ROOT}/training"
LOG_DIR="${PROJECT_ROOT}/training/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(basename "${CONFIG_FILE}" .yaml)_$(date +%Y%m%d_%H%M%S).log"

llamafactory-cli export "${CONFIG_FILE}" "$@" 2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}
[ $RC -eq 0 ] || { echo "[err] export failed rc=$RC"; exit $RC; }
# the export re-initializes a head it could not load without failing, so compare the exported head with the adapter's
read -r ADAPTER EXPORT < <(python -I - "${CONFIG_FILE}" "$@" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.update(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
print(c["adapter_name_or_path"], c["export_dir"])
PY
)
python "${PROJECT_ROOT}/training/check_merged_head.py" "${ADAPTER}" "${EXPORT}" || exit 1
echo "[ok] log: ${LOG_FILE}"
