import re
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from rna_sdb.datasets import DATASET_PATH
from rna_sdb.utils import db2pairs, read_fasta, write_fasta

RNA_SDB_PATH = DATASET_PATH / "rna_sdb"
RNA_SDB_PARSED_ALIGN_PATH = (
    RNA_SDB_PATH / "parsed_alignments" / "rfam_parsed_align.csv.gz"
)


class RNASDB:
    PRESPLITS = {
        "split_1": ("test_split_1.lst", "train_split_1.csv.gz"),  # (test, train)
        "split_2": ("test_split_2.lst", "train_split_2.csv.gz"),
        "split_3": ("test_split_3.lst", "train_split_3.csv.gz"),
        "split_4": ("test_split_4.lst", "train_split_4.csv.gz"),
        "split_5": ("test_split_5.lst", "train_split_5.csv.gz"),
        "split_6": ("test_split_6.lst", "train_split_6.csv.gz"),
        "split_7": ("test_split_7.lst", "train_split_7.csv.gz"),
        "split_8": ("test_split_8.lst", "train_split_8.csv.gz"),
        "split_9": ("test_split_9.lst", "train_split_9.csv.gz"),
    }

    def __init__(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        seed_only: bool = False,
        split_train_val: bool = True,
    ):
        self.seed_only = seed_only
        self.split_train_val = split_train_val

        # Parse dot-bracket notation to pair indices
        if "pair_indices" not in train_set.columns:
            train_set["pair_indices"] = self._process_db_structure(train_set)
        if "pair_indices" not in test_set.columns:
            test_set["pair_indices"] = self._process_db_structure(test_set)

        self.df_test = test_set

        if self.seed_only:
            print("Filtering training set to only include seed sequences")
            print(f"Original training set size: {len(train_set)}")
            train_set = train_set[train_set["seed"]]
            print(f"Filtered training set size: {len(train_set)}")

        if self.split_train_val:
            self.df_train, self.df_val = self.validation_split(train_set)
        else:
            self.df_train = train_set
            self.df_val = None

        self.max_seq_len = None  # need to call filter_seq_len to set this

    def validation_split(
        self,
        df_train: pd.DataFrame,
        val_ratio: float = 0.10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split df_train into training and validation sets.

        Validation set is constructed by sampling Rfam families from df_train until
        the validation set has at least the desired number of sequences.

        Args:
            df_train (pd.DataFrame): Training set
            val_ratio (float, optional): Ratio of validation set. Defaults to 0.1.
            random_state (Optional[int], optional): Random state for reproducibility.
                Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets
        """
        df_train_size = len(df_train)

        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError("val_ratio must be between 0 and 1 exclusive")

        val_size = int(len(df_train) * val_ratio)
        df_val = df_train.sample(0)

        # Sample from the training set until the validation set has at least 'val_size'
        while len(df_val) < val_size:
            families = df_train["rfam_family"].unique()
            # TODO: figure out how to handle random seed
            sample_family = np.random.choice(families, 1)[0]

            df_sample = df_train[df_train["rfam_family"] == sample_family]
            df_val = pd.concat([df_val, df_sample])

            df_train = df_train.drop(df_sample.index)

        assert (
            set(df_train["seq_id"].unique()) & set(df_val["seq_id"].unique()) == set()
        )
        assert len(df_train) + len(df_val) == df_train_size

        print(
            f"Train and validation split: {len(df_train)}, {len(df_val)} "
            f"({len(df_val) / (len(df_train) + len(df_val)):.1%})"
        )

        return df_train, df_val

    def filter_seq_len(self, max_seq_len: int):
        """Filter out sequences that are longer than the maximum sequence length.

        Args:
            max_seq_len (int): Maximum sequence length
        """

        print(f"Filtered sequences longer than {max_seq_len} nucleotides: ")

        df_train = self.df_train[self.df_train["seq"].str.len() <= max_seq_len]
        print(
            f"Train: {len(self.df_train) - len(df_train)} (out of {len(self.df_train)})"
        )

        if self.df_val is not None:
            df_val = self.df_val[self.df_val["seq"].str.len() <= max_seq_len]
            print(f"Val: {len(self.df_val) - len(df_val)} (out of {len(self.df_val)})")
        else:
            df_val = None
            print("No validation set, so skipped")

        df_test = self.df_test[self.df_test["seq"].str.len() <= max_seq_len]
        print(f"Test: {len(self.df_test) - len(df_test)} (out of {len(self.df_test)})")

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.max_seq_len = max_seq_len

    @staticmethod
    def load_parsed_align(
        file_path: Union[str, Path] = RNA_SDB_PARSED_ALIGN_PATH,
        drop_multi_hits: bool = True,
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path, compression="gzip")

        # Filter out sequences belonging to multiple families. This is because
        # some sequences have ambiguous family assignments. To minimize data
        # leakage, we remove these sequences.
        if drop_multi_hits:
            df = df.drop_duplicates(["seq_id", "seq"], keep=False)

        return df

    @classmethod
    def initialize_presplit(
        cls, split_name: str, seed_only: bool = False, split_train_val: bool = True
    ):
        # TODO: refactor, especially how split_train_val is being handled

        if split_name not in cls.PRESPLITS:
            raise ValueError(
                f"Invalid split name: {split_name}, "
                f"must be one of {cls.PRESPLITS.keys()}"
            )

        print(f"Initializing RNASDB with split: {split_name}")

        # Check if cached dataframes exist
        train_cache = RNA_SDB_PATH / f"{split_name}_cache_train.pq"
        test_cache = RNA_SDB_PATH / f"{split_name}_cache_test.pq"

        if train_cache.exists() and test_cache.exists():
            print(f"Loading cached dataframes: {train_cache}, {test_cache}")

            # pair_indices are saved as list string literals, so need to convert back
            df_train = pd.read_parquet(train_cache)
            df_test = pd.read_parquet(test_cache)

            return cls(
                df_train, df_test, seed_only=seed_only, split_train_val=split_train_val
            )

        # If there is no cached dataframes, load the data and cache them
        test_file, train_file = cls.PRESPLITS[split_name]

        with open(RNA_SDB_PATH / test_file) as f:
            test_split = f.read().splitlines()  # list of test Rfam families

        df_train_split = pd.read_csv(RNA_SDB_PATH / train_file)

        df_rnasdb = cls.load_parsed_align()

        df_train = df_rnasdb.merge(df_train_split, on="seq_id", how="inner")
        assert len(df_train) == len(
            df_train_split
        ), "Not all training sequences found in parsed alignments"

        df_test = df_rnasdb[df_rnasdb["rfam_family"].isin(test_split)]
        assert (
            len(df_test[df_test["seq_id"].isin(df_train["seq_id"])]) == 0
        ), "Test sequences found in training set"

        # __init__ to trigger processing
        rnasdb = cls(df_train, df_test, split_train_val=False)

        # Cache processed dataframes for faster loading
        assert len(rnasdb.df_train) == len(df_train) and rnasdb.df_val is None
        rnasdb.df_train.to_parquet(train_cache, index=False)
        rnasdb.df_test.to_parquet(test_cache, index=False)

        # __init__ again in case of seed_only
        rnasdb = cls(
            df_train, df_test, seed_only=seed_only, split_train_val=split_train_val
        )

        return rnasdb

    @staticmethod
    def _process_db_structure(df):
        return df.apply(lambda row: list(zip(*db2pairs(row["db_structure"]))), axis=1)


def process_clusters(cluster_path: Union[str, Path]) -> dict:
    """Process a clstr file generated from cd-hit-est or cd-hit-est-2d.

    Args:
        cluster_path (Union[str, Path]): Path to the cluster file

    Returns:
        dict: A dictionary of clusters where the key is the cluster id and the value
            is a list of tuples containing the sequence id, a boolean indicating
            if the sequence is the seed, and the similarity score.
    """
    clusters = dict()
    with open(cluster_path) as f:
        cluster_id = None
        for i, line in enumerate(f):
            if line.startswith(">Cluster"):  # cluster line
                cluster_id = int(re.match(r">Cluster\s+(\d+)", line).group(1))
                clusters[cluster_id] = []
            else:  # sequences in a cluster
                if cluster_id is None:
                    raise ValueError("The file must start with cluster line")

                match = re.match(r"(\d+)\s+\d+nt, >(.+)\.\.\. (.+)", line)
                if match is None:
                    raise ValueError(
                        f"Line {i} does not match expected format for sequences: {line}"
                    )

                seq_idx, seq_id, similarity_str = match.groups()

                cluster_seqs = clusters[cluster_id]

                assert int(seq_idx) == len(
                    cluster_seqs
                ), f"Expected sequence index {len(cluster_seqs)} but got {seq_idx}"

                if similarity_str == "*":  # the seed sequence
                    seed = True
                    similarity = 100.0
                else:
                    seed = False
                    similarity = float(
                        re.match(r"at [\+\-]/(\d+\.\d+)%", similarity_str).group(1)
                    )

                cluster_seqs.append((seq_id, seed, similarity))

    return clusters


def filter_similar_seqs_intra(
    seq_fasta: str,
    c: float = 0.8,
    n: int = 5,
    s: float = 0.0,
    num_threads: int = 4,
    print_output: bool = False,
) -> Tuple[list[Tuple[str, str]], dict]:
    """Filter similar sequences within a fasta file using cd-hit-est.

    Args:
        seq_fasta (str): Path to the sequence fasta file
        c (float, optional): Sequence identity threshold. Defaults to 0.8.
        n (int, optional): Word length. Defaults to 5.
        s (float, optional): Length difference cutoff. Defaults to 0.0.
            If set to 0.9, the shorter sequences need to be at least 90% length of
            the representative of the cluster
        num_threads (int, optional): Number of threads. Defaults to 4.
        print_output (bool, optional): Print the output of the cd-hit-est command.
            Defaults to False.

    Returns:
        list[Tuple[str, str]]: List of filtered sequences where each tuple contains
            the sequence id and the sequence.
        dict: A dictionary of clusters where the key is the cluster id and the value
            is a list of tuples containing the sequence id, a boolean indicating
            if the sequence is the seed, and the similarity score.
    """
    out_fasta = seq_fasta + f".filtered_{int(c * 100)}_{n}"
    cmd = [
        "cd-hit-est",
        "-i",
        seq_fasta,
        "-o",
        out_fasta,
        "-c",
        str(c),
        "-n",
        str(n),
        "-s",
        str(s),
        "-T",
        str(num_threads),
        "-M",
        "64000",
        "-d",
        "0",  # disable filtering of description by length in clstr file
    ]
    if print_output:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    seqs = read_fasta(out_fasta)

    clusters = process_clusters(out_fasta + ".clstr")

    return seqs, clusters


def filter_similar_seqs_inter(
    seq_fasta: str,
    seqs_to_filter_fasta: str,
    c: float = 0.8,
    n: int = 5,
    num_threads: int = 32,
    print_output: bool = False,
) -> list[Tuple[str, str]]:
    """Filter similar sequences between two fasta files using cd-hit-est-2d.

    Args:
        seq_fasta (str): Path to the sequence fasta file
        seqs_to_filter_fasta (str): Path to the sequence fasta file to filter out
        c (float, optional): Sequence identity threshold. Defaults to 0.8.
        n (int, optional): Word length. Defaults to 5.
        num_threads (int, optional): Number of threads. Defaults to 32.
        print_output (bool, optional): Print the output of the cd-hit-est-2d command.
            Defaults to False.

    Returns:
        list[Tuple[str, str]]: List of filtered sequences where each tuple contains
            the sequence id and the sequence.
    """
    out_fasta = seq_fasta + f".filtered_{int(c * 100)}_{n}"
    cmd = [
        "cd-hit-est-2d",
        "-i2",
        seq_fasta,
        "-i",
        seqs_to_filter_fasta,
        "-o",
        out_fasta,
        "-c",
        str(c),
        "-n",
        str(n),
        "-T",
        str(num_threads),
        "-M",
        "64000",
    ]
    if print_output:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    seqs = read_fasta(out_fasta)

    return seqs


def train_test_filter(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    c: float = 0.8,
    n: int = 5,
    num_threads: int = 4,
    print_output: bool = False,
) -> pd.DataFrame:
    """Filter out sequences in the training set that are similar to the test set.

    df_train and df_test DataFrames need to have 'seq_id' and 'seq' columns.

    Args:
        df_train (pd.DataFrame): Training set
        df_test (pd.DataFrame): Test set
        c (float, optional): Sequence identity threshold. Defaults to 0.8.
        n (int, optional): Word length. Defaults to 5.
        num_threads (int, optional): Number of threads. Defaults to 4.
        print_output (bool, optional): Print the output of the cd-hit-est-2d command.

    Returns:
        pd.DataFrame: Filtered training set
    """
    with tempfile.NamedTemporaryFile() as fasta_train, tempfile.NamedTemporaryFile() as fasta_test:  # noqa
        write_fasta(
            df_train["seq"].tolist(), fasta_train.name, df_train["seq_id"].tolist()
        )
        write_fasta(
            df_test["seq"].tolist(), fasta_test.name, df_test["seq_id"].tolist()
        )

        seqs = filter_similar_seqs_inter(
            fasta_train.name,
            fasta_test.name,
            c=c,
            n=n,
            num_threads=num_threads,
            print_output=print_output,
        )

        seq_ids = list(map(lambda x: x[0], seqs))
        df_train_filtered = df_train[df_train["seq_id"].isin(seq_ids)]

    print(
        f"Number of sequences in filtered training set: {len(df_train_filtered)}"
        f" (number filtered {len(df_train) - len(df_train_filtered)})"
    )

    return df_train_filtered


def train_cluster(
    df_train: pd.DataFrame,
    c: float = 0.8,
    n: int = 5,
    s: float = 0.0,
    num_threads: int = 4,
    print_output: bool = False,
    display_progbar: bool = True,
):
    """Cluster the training sequences and assign cluster metadata to the sequences.

    df_train needs to have 'seq_id' and 'seq' columns. The returned DataFrame will have
    additional columns with cluster metadata: 'cluster_id', 'cluster_size', 'seed',
    'similarity'.

    Args:
        df_train (pd.DataFrame): DataFrame containing training sequences
        c (float, optional): Sequence identity threshold. Defaults to 0.8.
        n (int, optional): Word length. Defaults to 5.
        s (float, optional): Length difference cutoff. Defaults to 0.0.
            If set to 0.9, the shorter sequences need to be at least 90% length of
            the representative of the cluster
        num_threads (int, optional): Number of threads to use. Defaults to 4.
        print_output (bool, optional): Print the output of the cd-hit-est command.
        display_progbar (bool, optional): Display progress bar. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing training sequences with cluster metadata
    """
    with tempfile.NamedTemporaryFile() as fasta_train:
        write_fasta(
            df_train["seq"].tolist(), fasta_train.name, df_train["seq_id"].tolist()
        )

        _, clusters = filter_similar_seqs_intra(
            fasta_train.name,
            c=c,
            n=n,
            s=s,
            num_threads=num_threads,
            print_output=print_output,
        )

    # Process 'clusters' to add columns for cluster metadata:
    #   'cluster_id', 'cluster_size', 'seed', 'similarity'
    df_cluster_data = []
    print("Assigning cluster metadata to training sequences...")
    for cluster_id, cluster_seqs in tqdm(clusters.items(), disable=not display_progbar):
        cluster_size = len(cluster_seqs)
        for seq_id, seed, similarity in cluster_seqs:
            df_cluster_data.append((seq_id, cluster_id, cluster_size, seed, similarity))
    df_cluster = pd.DataFrame(
        df_cluster_data,
        columns=["seq_id", "cluster_id", "cluster_size", "seed", "similarity"],
    )

    assert (
        len(df_cluster)
        == len(df_train)
        == len(df_cluster["seq_id"].unique())
        == len(df_train["seq_id"].unique())
    )
    assert set(df_cluster["seq_id"].unique()) == set(df_train["seq_id"].unique())

    df_train = df_train.merge(df_cluster, on="seq_id")
    assert len(df_train) == len(df_cluster)

    df_train["cluster_id"] = df_train["cluster_id"].astype(int)
    df_train["cluster_size"] = df_train["cluster_size"].astype(int)

    return df_train


def process_training_split(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_orthogonal_tests: dict[str, pd.DataFrame],
    c: float = 0.8,
    n: int = 5,
    num_cpus: int = 4,
    display_cdhit: bool = False,
) -> pd.DataFrame:
    """Process the training split by filtering out sequences similar to the test split
    and orthogonal test splits. Also, provide cluster metadata to the filtered training
    set

    Args:
        df_train (pd.DataFrame): Training set
        df_test (pd.DataFrame): Test set
        df_orthogonal_tests (dict[str, pd.DataFrame]): Dictionary of orthogonal test
            sets where the key is the dataset name and the value is the DataFrame
            containing the test set
        c (float, optional): Sequence identity threshold. Defaults to 0.8.
        n (int, optional): Word length. Defaults to 5.
        num_cpus (int, optional): Number of CPUs to use. Defaults to 4.
        display_cdhit (bool, optional): Display the output of cd-hit commands.

    Returns:
        pd.DataFrame: Filtered training set
    """
    print("Processing training split...")

    for dataset_name, df_orthogonal_test in df_orthogonal_tests.items():
        print(f"Filtering training set using {dataset_name}...")
        df_train_filtered = train_test_filter(
            df_train,
            df_orthogonal_test,
            c=c,
            n=n,
            num_threads=num_cpus,
            print_output=display_cdhit,
        )

    print("Filtering training set using test split...")
    df_train_filtered = train_test_filter(
        df_train_filtered,
        df_test,
        c=c,
        n=n,
        num_threads=num_cpus,
        print_output=display_cdhit,
    )

    print("Clustering training sequences...")
    df_train_clustered_list = []
    for _, df_family in df_train_filtered.groupby("rfam_family"):
        df_train_clustered = train_cluster(df_family, c=c, n=n, num_threads=num_cpus)
        df_train_clustered_list.append(df_train_clustered)
    df_train_clustered = pd.concat(df_train_clustered_list)

    assert len(df_train_filtered) == len(df_train_clustered)

    return df_train_clustered
