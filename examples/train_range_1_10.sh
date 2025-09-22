#!/usr/bin/env bash
set -euo pipefail

# 仅使用标签名排序后的第 1~10 个样本进行训练
# 可通过环境变量覆盖默认轮数/批大小：EPOCHS、BATCH_SIZE

export LABEL_RANGE="1-100"
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-30}"

# Choose python from virtualenv if present; allow override via PY
PY=${PY:-}
if [[ -z "${PY}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PY=python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    echo "No python interpreter found (tried .venv/bin/python, python, python3)" >&2
    exit 127
  fi
fi

"${PY}" -u train.py
