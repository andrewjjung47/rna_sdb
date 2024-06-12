import argparse

import pandas as pd

from rna_sdb.datasets import RNASDB
from rna_sdb.dp_methods import eval_metrics, evaluate_dp_method


def run(
    methods,
    num_processes: int = 1,
    chunk_size: int = 1,
    max_seq_len: int = 450,
    save: bool = False,
    load_saved: bool = False,
):
    pred_col = "struct_pred"

    for method in methods:
        print(f"Running evaluation for {method}")

        splits = RNASDB.PRESPLITS.keys()

        for split in splits:
            save_file = f"{method}_rnasdb_{split}.csv.gz"

            if load_saved:
                print(f"Loading saved predictions from {save_file}")
                df = pd.read_csv(save_file)
                agg_metrics = eval_metrics(df, df[pred_col].tolist())
            else:
                rnasdb = RNASDB.initialize_presplit(split)
                if max_seq_len is not None:
                    rnasdb.filter_seq_len(max_seq_len)

                df = rnasdb.df_test
                db_struct_preds, agg_metrics = evaluate_dp_method(
                    method,
                    df,
                    num_processes=num_processes,
                    chunk_size=chunk_size,
                )
                df["struct_pred"] = db_struct_preds

            print(f"Aggregated metrics for {split}: {agg_metrics}:")

            if save and not load_saved:
                print(f"Saving predictions to {save_file}")
                df.to_csv(save_file, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating DP-based methods"
    )
    parser.add_argument(
        "--methods",
        help="Select a method to evaluate (default runs evaluation for all methods)",
        choices=[
            "rnafold",
            "rnastructure",
            "contrafold",
            # "linearfold_v",
            # "linearfold_c",
        ],
        nargs="+",
        default=[
            "rnafold",
            "rnastructure",
            "contrafold",
            # "linearfold_v",
            # "linearfold_c",
        ],
    )
    parser.add_argument(
        "--num_processes",
        help="Number of processes to use for evaluation",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk_size",
        help="Size of chunks to use for evaluating in multi processes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_seq_len",
        help="Maximum sequence length to consider for evaluation",
        type=int,
        default=450,
    )
    parser.add_argument(
        "--save", help="Save the predictions to a csv", action="store_true"
    )
    parser.add_argument(
        "--load_saved", help="Load the saved predictions", action="store_true"
    )
    args = parser.parse_args()

    run(
        args.methods,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size,
        max_seq_len=args.max_seq_len,
        save=args.save,
        load_saved=args.load_saved,
    )
