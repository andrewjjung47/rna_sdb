import argparse
import gzip
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from rna_sdb.datasets import RFAM_PATH, RNA_SDB_PATH

# Each training split must exclude the following test families:
test_splits = dict()
# Split 1 -- test families: tRNA and tmRNA
test_splits["split_1"] = [
    "RF00005",
    "RF01852",
    "RF00023",
    "RF02544",
    "RF01851",
    "RF01850",
    "RF01849",
    "RF00233",
    "RF01075",
    "RF01077",
    "RF01084",
    "RF01085",
    "RF01088",
    "RF01101",
]
# Split 2 -- test families: SRP RNA
test_splits["split_2"] = [
    "RF00169",
    "RF01854",
    "RF00017",
    "RF01855",
    "RF01857",
    "RF01502",
    "RF01856",
    "RF01570",
    "RF04183",
]
# Split 3 -- test families: telomerase RNA
test_splits["split_3"] = ["RF01050", "RF00024", "RF00025", "RF02462"]
# Split 4 -- test families: 5S rRNA
test_splits["split_4"] = ["RF00001", "RF02547", "RF02555", "RF02554"]
# Split 5 -- test families: RNase P
test_splits["split_5"] = [
    "RF00010",
    "RF00009",
    "RF00011",
    "RF00373",
    "RF02357",
    "RF00030",
    "RF01577",
]
# Split 6 -- test families: Group 1 and 2 introns
test_splits["split_6"] = [
    "RF01998",
    "RF01999",
    "RF02001",
    "RF02003",
    "RF02004",
    "RF02005",
    "RF02012",
    "RF00029",
    "RF00028",
]
# Split 7 -- test families: 23S rRNA
test_splits["split_7"] = ["RF00002", "RF02540", "RF02541", "RF02543", "RF02546"]
# Split 8 -- test families: 13S rRNA
test_splits["split_8"] = ["RF01959", "RF00177", "RF01960", "RF02542", "RF02545"]


def load_rfamseq_stats() -> pd.DataFrame:
    """Load statistics for Rfamseq.

    Returns:
        pd.DataFrame: statistics for Rfamseq with columns:
            - rfam_family: Rfam family ID
            - seq_count: number of sequences in the family
            - noncanonical_count: number of sequences with noncanonical bases
    """
    print("Loading Rfamseq...")

    rfamseq_path = RFAM_PATH / "rfamseq"

    seq_dist = []  # for Rfamseq statistics
    bases = set()  # set of all bases in the dataset

    for filename in tqdm(rfamseq_path.glob("RF*.fa.gz")):
        rfam_family = filename.name.replace(".fa.gz", "")

        seq_count = 0  # number of sequences
        noncanonical_count = 0  # number of sequences with noncanonical bases
        with gzip.open(filename, "rt") as f:
            for line in f:
                assert line.startswith(">"), "Expected FASTA description line first"

                line = next(f).strip().upper()  # sequence line
                assert re.match(
                    r"^[A-Z]+$", line
                ), f"Invalid character in sequence line: {line}"

                seq_count += 1

                if re.search(r"[^ACGTU]", line):  # canonical bases: A, C, G, U, T
                    noncanonical_count += 1

                bases.update(line)

        seq_dist.append((rfam_family, seq_count, noncanonical_count))

    df = pd.DataFrame(
        seq_dist, columns=["rfam_family", "seq_count", "noncanonical_count"]
    )
    df = df.sort_values("rfam_family").reset_index(drop=True)
    df["canonical_count"] = df["seq_count"] - df["noncanonical_count"]

    seq_count = df["seq_count"].sum()
    canonical_count = df["canonical_count"].sum()
    noncanonical_count = df["noncanonical_count"].sum()
    print("Rfamseq loaded. Statistics:")
    print(f"Total of sequences: {seq_count:,}")
    print(
        f"Number of sequences with non-canonical bases: {noncanonical_count:,} "
        f"({noncanonical_count / seq_count:.1%})"
    )
    print(
        f"Number of sequences with canonical bases: {canonical_count} "
        f"({(canonical_count) / seq_count:.1%})"
    )
    print(f"Bases occuring in the dataset: {bases} ({len(bases)} bases)")
    print(f"Number of families: {len(df):,}")

    return df


def split_rfamseq(
    df: pd.DataFrame,
    test_include: Optional[set[str]] = set(),
    test_exclude: Optional[set[str]] = set(),
    num_splits: int = 9,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split Rfamseq into training and test sets.
    Can specify which Rfam families to include/exclude from the test set.
    For simplicity, only specify either test_include or train_exclude.

    Args:
        df (pd.DataFrame): statistics for Rfamseq (from load_rfamseq_stats())
        test_include (list[str], optional): list of Rfam families to include
                                            in the test set. Defaults to None.
        test_exclude (list[str], optional): list of Rfam families to exclude
                                    from the training set. Defaults to None.
        num_splits (int, optional): number of splits which will be used to
                                    determine test set size. Defaults to 9.
        random_state (int, optional): random seed for reproducibility of splitting.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: training and test sets
    """
    if test_include.intersection(test_exclude):
        raise ValueError("test_include and test_exclude must be disjoint sets")

    print("Splitting Rfamseq...")

    print("Statistics before balancing:")
    df_test = df[df["rfam_family"].isin(test_include)]
    df_train = df[~df["rfam_family"].isin(test_include)]

    print(f"Number of train families: {len(df_train):,}")
    print(f"Number of test families: {len(df_test):,}")
    train_ratio = df_train["canonical_count"].sum() / df["canonical_count"].sum()
    print(f"Train ratio: {train_ratio:.1%}")

    # Balance the training / test sets
    target_test_size = df["canonical_count"].sum() / num_splits
    while df_test["canonical_count"].sum() < target_test_size:
        sample_family = df_train.sample(1, random_state=random_state)

        if sample_family["rfam_family"].values[0] in test_exclude:
            continue

        df_test = pd.concat([df_test, sample_family])
        df_train = df_train.drop(sample_family.index)

    print("Statistics after balancing:")
    print(f"Number of train families: {len(df_train):,}")
    print(f"Number of test families: {len(df_test):,}")
    train_ratio = df_train["canonical_count"].sum() / df["canonical_count"].sum()
    print(f"Train ratio: {train_ratio:.1%}")

    return df_train, df_test


def run(min_seq_count: int = 20, random_state: Optional[int] = None):
    if random_state is not None:
        random_state = np.random.RandomState(random_state)

    df_rfamseq_stats = load_rfamseq_stats()
    df_rfamseq_stats.to_csv(RNA_SDB_PATH / "rfamseq_stats.csv", index=False, sep="\t")

    # Filter families with less than min_seq_count of canonical sequences and
    # reserve them for testing
    df_rfamseq_filtered_stats = df_rfamseq_stats[
        df_rfamseq_stats["canonical_count"] < min_seq_count
    ]
    test_reserved = set(df_rfamseq_filtered_stats["rfam_family"])

    num_seq = df_rfamseq_stats["canonical_count"].sum()
    num_seq_filtered = df_rfamseq_filtered_stats["canonical_count"].sum()
    print(f"Total number of Rfam families reserved for testing: {len(test_reserved)}")
    print(
        f"Total of sequences in reserved for testing: {num_seq_filtered:,} "
        f"({num_seq_filtered / num_seq:.1%})"
    )

    # Split Rfamseq into training and test sets
    for split, test_include in test_splits.items():
        print(f"\nSplit {split}")
        _, df_test = split_rfamseq(
            df_rfamseq_stats,
            test_include=set(test_include) | test_reserved,
            random_state=random_state,
        )

        # Save families in the test set
        with open(RNA_SDB_PATH / f"test_{split}.lst", "w") as file:
            for item in df_test["rfam_family"]:
                file.write(f"{item}\n")

    # Last split will have none of the previous test families in the test set
    print("\nLast split")
    test_exclude = set()
    for split, test_include in test_splits.items():
        test_exclude.update(test_include)

    _, df_test = split_rfamseq(
        df_rfamseq_stats,
        # Some families in test_exclude have less than 20 sequences and have
        # overlaps with test_reserved. We will exclude them from the test set
        # because we want the last test split to not include of the previous test
        # families, even though some of them have less than 20 sequences.
        test_include=test_reserved - test_exclude,
        test_exclude=test_exclude,
        random_state=random_state,
    )
    # Save families in the test set
    with open(RNA_SDB_PATH / f"test_split_{len(test_splits) + 1}.lst", "w") as file:
        for item in df_test["rfam_family"]:
            file.write(f"{item}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Rfamseq dataset")
    parser.add_argument(
        "--min_seq_count",
        type=int,
        default=20,
        help="Minimum number of canonical sequences required for a family",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random seed for reproducibility of splitting",
    )
    args = parser.parse_args()

    run(min_seq_count=args.min_seq_count, random_state=args.random_state)
