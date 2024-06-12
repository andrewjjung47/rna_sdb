import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from rna_sdb.data import idx2matrix
from rna_sdb.utils import db2pairs


def load_archiveII_pred(pred_path: Path, label_path: Path):
    df_pred = pd.read_parquet(pred_path)
    df_label = pd.read_parquet(label_path)

    assert (df_pred["seq"] == df_label["seq"]).all()

    return df_pred, df_label


def parse_bpseq(bpseq):
    pair_indices = ([], [])
    for idx, pair_idx in bpseq:
        if pair_idx != 0:  # 0 means no pair
            # 1-indexed to 0-indexed
            pair_indices[0].append(idx - 1)
            pair_indices[1].append(pair_idx - 1)

    return pair_indices


def eval_metrics(df_pred: pd.DataFrame, df_label: pd.DataFrame):
    """Evaluate metrics for a given set of predictions.

    Calculate F1, precision, and recall.
    """
    assert "bp_matrix" in df_pred.columns and "pair_indices" in df_label.columns

    metrics = MetricCollection([BinaryF1Score(), BinaryPrecision(), BinaryRecall()])

    for (_, row_1), (_, row_2) in zip(df_pred.iterrows(), df_label.iterrows()):
        seq = row_2["seq"]

        pair_indices = parse_bpseq(row_1["bpseq"])
        y_pred = torch.Tensor(idx2matrix(pair_indices, len(seq)))
        y_label = torch.Tensor(idx2matrix(row_2["pair_indices"], len(seq)))

        metrics.update(y_pred, y_label)

    agg_metrics = metrics.compute()

    print(agg_metrics)

    return agg_metrics


def run(pred_path: str, label_path: str):
    pred_path = Path(pred_path)
    label_path = Path(label_path)

    df_pred, df_label = load_archiveII_pred(pred_path, label_path)

    eval_metrics(df_pred, df_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MXFold2 predictions")
    parser.add_argument("--pred_path", type=str, help="Path to the predictions file")
    parser.add_argument(
        "--label_path",
        type=str,
        help="Path to the labels file",
    )
    args = parser.parse_args()

    run(args.pred_path, args.label_path)
