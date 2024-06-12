import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from rna_sdb.data import RNASDBDatasetModule

SAMPLES_PER_EPOCH = 10784


def generate_weighted_df(
    dataset: RNASDBDatasetModule, num_epochs: int = 10
) -> pd.DataFrame:
    df_train = dataset.rnasdb.df_train
    weights = dataset._calc_sampling_weights(df_train)
    sampler = WeightedRandomSampler(
        weights, SAMPLES_PER_EPOCH * num_epochs, replacement=True
    )

    df_sample = df_train.iloc[list(sampler)]

    return df_sample


def run(
    split_name: str, output_dir: Optional[str] = None, subsample: Optional[float] = None
):
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    if not 0.0 < subsample < 1.0:
        raise ValueError("Subsample must be in the range (0, 1)")

    dataset = RNASDBDatasetModule(batch_size=1, split_name=split_name)
    dataset.setup("test")  # call 'test' since no need for train / val split

    df = generate_weighted_df(dataset)
    df.to_parquet(output_dir / f"mxfold2_rnasdb_weighted_{split_name}.pq")

    # Generate test data
    df_test = dataset.rnasdb.df_test
    df_archiveII = dataset.df_archiveII

    if subsample is not None:
        df_test = df_test.sample(frac=subsample)

    subsample_path_name = "" if subsample is None else f"_{subsample}"
    output_path = (
        output_dir / f"mxfold2_rnasdb_test_{split_name}{subsample_path_name}.pq"
    )
    df_test.to_parquet(output_path)
    df_archiveII.to_parquet(output_dir / f"mxfold2_rnasdb_archiveII_{split_name}.pq")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MXFold2 weighted sampled datasets"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        help="Which split to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output (default is the directory of this file)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="How much to subsample the RNASDB test set "
        "(0.0 < subsample < 1.0). Default is None (no subsampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for weighted random sampling and pandas subsampling",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    run(args.split_name, output_dir=args.output_dir, subsample=args.subsample)
