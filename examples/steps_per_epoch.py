#!/usr/bin/env python3
"""
计算当前数据集的样本数与每轮(steps/epoch)。

与 train.py 的采样规则一致：
- 仅统计 ./labels 下的 .txt，并要求 ./images 下存在同名图片(.png/.jpg/.jpeg)。
- 支持环境变量：
  - LABEL_RANGE="a-b"（1 基，按标签文件名排序后取区间）
  - LIMIT_N=N（仅取前 N 个）
  - BATCH_SIZE（默认 30）
"""

from __future__ import annotations

import math
import os
from typing import List


def list_samples() -> List[str]:
    labels_dir = './labels'
    images_dir = './images'
    data = []
    if not os.path.isdir(labels_dir):
        return data
    for name in os.listdir(labels_dir):
        if not name.lower().endswith('.txt'):
            continue
        base = os.path.splitext(name)[0]
        candidates = [
            os.path.join(images_dir, base + ext)
            for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        ]
        if any(os.path.exists(p) for p in candidates):
            data.append(os.path.join(labels_dir, name))
    data.sort()

    lr = os.getenv('LABEL_RANGE')
    if lr:
        try:
            a, b = lr.strip().split('-', 1)
            a_i = max(1, int(a))
            b_i = int(b)
            data = data[a_i - 1 : b_i]
        except Exception:
            pass

    limit_n = os.getenv('LIMIT_N')
    if limit_n:
        try:
            n = int(limit_n)
            if n > 0:
                data = data[:n]
        except Exception:
            pass
    return data


def main() -> None:
    samples = list_samples()
    n = len(samples)
    bs = int(os.getenv('BATCH_SIZE', '30'))
    steps = math.ceil(n / bs) if n > 0 else 0
    print(f"samples={n}")
    print(f"batch_size={bs}")
    print(f"steps_per_epoch=ceil({n}/{bs})={steps}")


if __name__ == '__main__':
    main()

