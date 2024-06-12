import argparse
import multiprocessing as mp
from typing import Optional

import pandas as pd

from rna_sdb.datasets import RFAM_PATH, RNA_SDB_PATH, load_archiveII
from rna_sdb.datasets.rna_sdb import process_training_split
from rna_sdb.utils import read_fasta


def load_rfamseq(
    rfam_families: list[str],
    filter_non_canonical: bool = True,
    convert_u_to_t: bool = True,
    filter_duplicates: bool = True,
    filter_multi_hits: bool = True,
    cleanup_seq_id: bool = True,
) -> pd.DataFrame:
    """Load Rfamseq sequences for a given list of Rfam families

    Args:
        rfam_families (list[str]): List of Rfam families
        filter_non_canonical (bool, optional): Filter out non-canonical bases.
            Defaults to True.
        convert_u_to_t (bool, optional): Convert U to T. Defaults to True.
        filter_duplicates (bool, optional): Filter out duplicate sequences.
            Defaults to True.
        filter_multi_hits (bool, optional): Filter out sequences belonging to
            multiple families. Defaults to True.
        cleanup_seq_id (bool, optional): Cleanup sequence ID. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the Rfamseq. Contains columns:
            - seq_id: Sequence ID
            - seq: Sequence
            - rfam_family: Rfam family
    """
    rfamseq_path = RFAM_PATH / "rfamseq"

    df_list = []
    for family in rfam_families:
        file_path = rfamseq_path / f"{family}.fa.gz"

        seqs = read_fasta(file_path, filter_non_canonical, convert_u_to_t)

        df = pd.DataFrame(seqs, columns=["seq_id", "seq"])
        df["rfam_family"] = family
        df_list.append(df)

    df = pd.concat(df_list)

    if filter_duplicates:
        print("Filtering duplicates:")
        print(f"    - before filtering: {len(df)}")
        df = df.drop_duplicates()
        print(f"    - after filtering: {len(df)}")

    # Filter out sequences belonging to multiple families. This is because
    # some sequences have ambiguous family assignments. To minimize data
    # leakage, we remove these sequences.
    if filter_multi_hits:
        print("Filtering multiple hits:")
        print(f"    - before filtering: {len(df)}")
        df = df.drop_duplicates(["seq_id", "seq"], keep=False)
        print(f"    - after filtering: {len(df)}")

    assert not df.duplicated("seq_id").any(), "Duplicate sequence IDs found"

    # Cleanup sequence ID by removing meta data
    # e.g. CP022571.1/1542200-1542289 Prosthecochloris sp. GSB1, complete genome.
    #       becomes CP022571.1/1542200-1542289
    if cleanup_seq_id:
        df["seq_id"] = df["seq_id"].apply(lambda x: x.split()[0])

    return df


def run(num_cpus: Optional[int] = mp.cpu_count(), display_cdhit: bool = False):
    print(f"Number of CPUs: {num_cpus}")

    # Load Rfamseq
    df_rfamseq_stats = pd.read_csv(RNA_SDB_PATH / "rfamseq_stats.csv", sep="\t")
    df_rfamseq = load_rfamseq(df_rfamseq_stats["rfam_family"].tolist())

    # Load orthogonal test set: ArchiveII
    df_archiveII = load_archiveII()

    for test_split_path in sorted(RNA_SDB_PATH.glob("test_split_*.lst")):
        print(f"Processing {test_split_path.stem}...")
        test_split = test_split_path.stem
        with open(test_split_path) as f:
            test_families = f.read().splitlines()

        train_families = df_rfamseq_stats[
            ~df_rfamseq_stats["rfam_family"].isin(test_families)
        ]["rfam_family"].tolist()

        df_train = df_rfamseq[df_rfamseq["rfam_family"].isin(train_families)]
        df_test = df_rfamseq[df_rfamseq["rfam_family"].isin(test_families)]

        print(f"Number of training sequences: {len(df_train)}")
        print(f"Number of test sequences: {len(df_test)}")

        df_archiveII_split = df_archiveII[df_archiveII["split"] == test_split]

        df_train_filtered = process_training_split(
            df_train,
            df_test,
            {"ArchiveII": df_archiveII_split},
            num_cpus=num_cpus,
            display_cdhit=display_cdhit,
        )

        # Save the training split
        train_split_path = (
            str(test_split_path).replace("test", "train").replace("lst", "csv.gz")
        )
        # Only save the necessary columns to reduce file size
        df_train_filtered = df_train_filtered[
            ["seq_id", "cluster_id", "cluster_size", "seed", "similarity"]
        ]
        df_train_filtered.to_csv(train_split_path, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process RNASDB training splits to filter out seqs from test sets"
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=mp.cpu_count(),
        help="Number of CPUs to use",
    )
    parser.add_argument(
        "--display_cdhit",
        action="store_true",
        help="Display CD-HIT output",
    )
    args = parser.parse_args()
    run(args.cpu, display_cdhit=args.display_cdhit)
