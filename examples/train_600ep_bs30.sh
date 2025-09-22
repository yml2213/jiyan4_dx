#!/usr/bin/env bash
set -euo pipefail

# 训练 600 轮，每批 30 张
EPOCHS=600 BATCH_SIZE=30 python -u train.py

