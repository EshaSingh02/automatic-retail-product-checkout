import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def compute_rpc_metrics(gt_counts, pred_counts):
    categories = set(gt_counts) | set(pred_counts)

    gt_total = sum(gt_counts.values())
    pred_total = sum(pred_counts.values())
    acd = abs(gt_total - pred_total)

    mccd_values = []
    for category in categories:
        gt = gt_counts.get(category, 0)
        pred = pred_counts.get(category, 0)
        if gt > 0:
            mccd_values.append(abs(pred - gt) / gt)

    mccd = float(np.mean(mccd_values)) if mccd_values else 0.0

    intersection = sum(min(gt_counts.get(c, 0), pred_counts.get(c, 0)) for c in categories)
    union = sum(max(gt_counts.get(c, 0), pred_counts.get(c, 0)) for c in categories)
    mciou = intersection / union if union else 0.0

    return acd, mccd, mciou


def main():
    parser = argparse.ArgumentParser(
        description="Utilities for RPC count-based metric evaluation."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    gt = json.loads(args.ground_truth.read_text())
    pred = json.loads(args.predictions.read_text())

    acd, mccd, mciou = compute_rpc_metrics(Counter(gt), Counter(pred))

    print(f"ACD: {acd:.4f}")
    print(f"mCCD: {mccd:.4f}")
    print(f"mCIoU: {mciou:.4f}")


if __name__ == "__main__":
    main()
