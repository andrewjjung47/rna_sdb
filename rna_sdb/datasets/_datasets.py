import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from rna_sdb.utils import read_bpseq, read_fasta

NONCANONICAL_BASES = r"[^ATGCU]"

DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets"
BPRNA_PATH = DATASET_PATH / "bprna"
ARCHIVEII_PATH = DATASET_PATH / "archiveII"
BPRNA_NEW_PATH = DATASET_PATH / "bprna_new"
RFAM_PATH = DATASET_PATH / "rfam"
RFAMSEQ_PATH = RFAM_PATH / "rfamseq"
RNA3DB_PATH = DATASET_PATH / "rna3db"

# Pre-determined splits of ARCHIVEII by family for testing
ARCHIVEII_SPLIT = {
    "tRNA": "test_split_1",
    "tmRNA": "test_split_1",
    "srp": "test_split_2",
    "telomerase": "test_split_3",
    "5s": "test_split_4",
    "RNaseP": "test_split_5",
    "grp1": "test_split_6",
    "grp2": "test_split_6",
    "23s": "test_split_7",
    "16s": "test_split_8",
}


def load_bprna(
    filter_non_canonical: bool = True,
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
) -> pd.DataFrame:
    """Load bpRNA dataset

    Args:
        filter_non_canonical (bool, optional): Whether to filter out non-canonical
                                                bases. Defaults to True.
        convert_to_upper (bool, optional): Convert sequence to uppercase.
            Defaults to True.
        convert_u_to_t (bool, optional): Convert U to T. Defaults to False.

    Returns:
        pd.DataFrame: bpRNA dataset with columns "seq_id", "seq", "split",
                        "pair_indices"
    """
    logging.info("Loading bpRNA dataset...")

    data_list = []
    logging.info(f"Loading data from {BPRNA_PATH}")
    for file in BPRNA_PATH.glob("*/*"):
        data_split = file.parent.stem
        assert data_split in ["TR0", "VL0", "TS0"], f"Unknown data split: {data_split}"

        seq_id = file.stem

        seq, pair_indices = read_bpseq(
            file, convert_to_upper=convert_to_upper, convert_u_to_t=convert_u_to_t
        )
        seq = seq.upper()

        data_list.append((seq_id, seq, data_split, pair_indices, str(file)))

    if len(data_list) == 0:
        raise FileNotFoundError(f"No data found in {BPRNA_PATH}")

    if len(data_list) == 0:
        raise FileNotFoundError(f"No data found in {BPRNA_PATH}")

    dataset = pd.DataFrame(
        data_list, columns=["seq_id", "seq", "split", "pair_indices", "path"]
    )

    logging.info(f"Loaded {len(dataset)} examples.")

    if filter_non_canonical:
        print(f"Before filtering non-canonical bases: {len(dataset)}")
        dataset = dataset[~dataset["seq"].str.match(NONCANONICAL_BASES)]
        print(f"After filtering non-canonical bases: {len(dataset)}")

    return dataset


def load_bprna_new(
    filter_non_canonical: bool = True,
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
) -> pd.DataFrame:
    """Load bpRNA-new dataset

    Args:
        filter_non_canonical (bool, optional): Whether to filter out non-canonical
                                                bases. Defaults to True.
        convert_to_upper (bool, optional): Convert sequence to uppercase.
            Defaults to True.
        convert_u_to_t (bool, optional): Convert U to T. Defaults to False.

    Returns:
        pd.DataFrame: bpRNA-new dataset with columns "seq_id", "seq", "pair_indices",
                        "path"
    """
    logging.info("Loading bpRNA-new dataset...")

    data_list = []
    logging.info(f"Loading data from {BPRNA_NEW_PATH}")
    for file_path in BPRNA_NEW_PATH.glob("*.bpseq"):
        match = re.match(r"^(.+)\.bpseq$", file_path.name)
        seq_id = match.group(1)

        seq, pair_indices = read_bpseq(
            file_path, convert_to_upper=convert_to_upper, convert_u_to_t=convert_u_to_t
        )

        data_list.append((seq_id, seq, pair_indices, str(file_path)))

    dataset = pd.DataFrame(data_list, columns=["seq_id", "seq", "pair_indices", "path"])

    logging.info(f"Loaded {len(dataset)} examples.")

    if filter_non_canonical:
        print(f"Before filtering non-canonical bases: {len(dataset)}")
        dataset = dataset[~dataset["seq"].str.match(NONCANONICAL_BASES)]
        print(f"After filtering non-canonical bases: {len(dataset)}")

    return dataset


def load_archiveII(
    split_name: Optional[str] = None,
    filter_non_canonical: bool = True,
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
    max_seq_len: Optional[int] = None,
) -> pd.DataFrame:
    """Load ArchiveII dataset

    Args:
        split_name (Optional[str], optional): Split name of the subset to load.
            If None, load the entire dataset. Defaults to None.
        filter_non_canonical (bool, optional): Whether to filter out non-canonical
                                                bases. Defaults to True.
        convert_to_upper (bool, optional): Convert sequence to uppercase.

        convert_u_to_t (bool, optional): Convert U to T. Defaults to False.

    Returns:
        pd.DataFrame: ArchiveII dataset with columns "seq_id", "seq", "pair_indices",
                        "family", 'split', "path"
    """
    logging.info("Loading ArchiveII dataset...")

    data_list = []
    logging.info(f"Loading data from {ARCHIVEII_PATH}")
    for file_path in ARCHIVEII_PATH.glob("*.bpseq"):
        match = re.match(r"^([a-zA-Z0-9]+)_(.+)\.bpseq$", file_path.name)
        family, seq_id = match.groups()
        split = ARCHIVEII_SPLIT[family]

        seq, pair_indices = read_bpseq(
            file_path, convert_to_upper=convert_to_upper, convert_u_to_t=convert_u_to_t
        )

        data_list.append(
            (
                # Combination of family and seq_id makes the seq_id unique
                f"{family}_{seq_id}",
                seq,
                pair_indices,
                family,
                split,
                str(file_path),
            )
        )

    if len(data_list) == 0:
        raise FileNotFoundError(f"No data found in {ARCHIVEII_PATH}")

    dataset = pd.DataFrame(
        data_list, columns=["seq_id", "seq", "pair_indices", "family", "split", "path"]
    )

    logging.info(f"Loaded {len(dataset)} examples.")

    if filter_non_canonical:
        print(f"Before filtering non-canonical bases: {len(dataset)}")
        dataset = dataset[~dataset["seq"].str.match(NONCANONICAL_BASES)]
        print(f"After filtering non-canonical bases: {len(dataset)}")

    if max_seq_len is not None:
        dataset = dataset[dataset["seq"].str.len() <= max_seq_len]

    if split_name is not None:
        dataset = dataset[dataset["split"] == split_name]

    return dataset


def load_rnastralign(
    filter_non_canonical: bool = True,
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
) -> pd.DataFrame:
    """Load RNAStrAlign dataset (only contain sequences)

    Files are obtained from https://github.com/ml4bio/e2efold

    Args:
        filter_non_canonical (bool, optional): Whether to filter out non-canonical
                                                bases. Defaults to True.
        convert_to_upper (bool, optional): Convert sequence to uppercase.
            Defaults to True.
        convert_u_to_t (bool, optional): Convert U to T. Defaults to False.

    Returns:
        pd.DataFrame: RNAStrAlign dataset with columns "seq_id", "seq", "split"
    """
    logging.info("Loading RNAStrAlign dataset...")

    files = [  # (file_name, split)
        ("rnastralign_train_no_redundant.seq.gz", "train"),
        ("rnastralign_val_no_redundant.seq.gz", "val"),
        ("rnastralign_test_no_redundant.seq.gz", "test"),
    ]

    df_list = []
    for file, split in files:
        logging.info(f"Loading data from {file}")
        seqs = read_fasta(
            DATASET_PATH / file,
            filter_non_canonical=filter_non_canonical,
            convert_to_upper=convert_to_upper,
            convert_u_to_t=convert_u_to_t,
        )

        df = pd.DataFrame(seqs, columns=["seq_id", "seq"])
        df["split"] = split

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    logging.info(f"Loaded {len(df)} examples.")

    return df


def load_rfamseq_fasta(
    rfam_family: str,
    filter_non_canonical: bool = False,
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
) -> pd.DataFrame:
    """Load Rfamseq sequences for a given Rfam family

    Args:
        rfam_family (str): Rfam family
        filter_non_canonical (bool, optional): Filter out non-canonical bases.
        Defaults to False.
        convert_to_upper (bool, optional): Convert sequence to uppercase.
        Defaults to True.
        convert_u_to_t (bool, optional): Convert U to T. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the Rfamseq. Contains columns:
            - seq_id: Sequence ID
            - seq: Sequence
            - rfam_family: Rfam family
    """
    file_path = RFAMSEQ_PATH / f"{rfam_family}.fa.gz"

    seqs = read_fasta(file_path, filter_non_canonical, convert_to_upper, convert_u_to_t)

    df = pd.DataFrame(seqs, columns=["seq_id", "seq"])
    df["rfam_family"] = rfam_family

    return df
