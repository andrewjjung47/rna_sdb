from typing import List, Literal, Optional

import pandas as pd
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from rna_sdb.data import idx2matrix
from rna_sdb.dp_methods import (
    run_contrafold,
    run_linearfold_c,
    run_linearfold_v,
    run_rnafold,
    run_rnastructure,
)
from rna_sdb.utils import db2pairs


def eval_metrics(df: pd.DataFrame, db_struct_preds: List[str]):
    """Evaluate metrics for a given set of predictions.

    Calculate F1, precision, and recall. Assume that the DataFrame has
    columns "seq" and "pair_indices".
    """
    assert "seq" in df.columns and "pair_indices" in df.columns

    metrics = MetricCollection([BinaryF1Score(), BinaryPrecision(), BinaryRecall()])

    for (_, row), struct_pred in zip(df.iterrows(), db_struct_preds):
        y_label = torch.Tensor(idx2matrix(row["pair_indices"], len(row["seq"])))

        one_idx = list(zip(*db2pairs(struct_pred)))

        y_pred = torch.Tensor(idx2matrix(one_idx, len(row["seq"])))

        metrics.update(y_pred, y_label)

    agg_metrics = metrics.compute()

    return agg_metrics


def evaluate_dp_method(
    method: Literal[
        "rnafold", "rnastructure", "contrafold", "linearfold_v", "linearfold_c"
    ],
    df: pd.DataFrame,
    num_processes: int = 1,
    chunk_size: int = 1,
    saved_preds: Optional[List[str]] = None,
):
    if method == "rnafold":
        print("Using RNAfold")
        method = run_rnafold
    elif method == "rnastructure":
        print("Using RNAstructure")
        method = run_rnastructure
    elif method == "contrafold":
        print("Using CONTRAfold")
        method = run_contrafold
    elif method == "linearfold_v":
        print("Using LinearFold with ViennaRNA parameters")
        method = run_linearfold_v
    elif method == "linearfold_c":
        print("Using LinearFold with CONTRAfold parameters")
        method = run_linearfold_c
    else:
        raise ValueError(f"Unknown method: {method}")

    assert "seq" in df.columns and "pair_indices" in df.columns

    # Run predictions
    seqs = df["seq"].tolist()

    if saved_preds:
        print("Using saved predictions")
        db_struct_preds = saved_preds
    else:
        db_struct_preds = method(seqs, num_processes, chunk_size, prog_bar=True)

    agg_metrics = eval_metrics(df, db_struct_preds)

    return db_struct_preds, agg_metrics
