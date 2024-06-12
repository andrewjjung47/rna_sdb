"""
Processes Rfam sequences by aligning to covariance models (CMs) using 'cmalign'
and parsing the alignment files to extract sequences and their structures.
"""

import argparse
import gzip
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

from rna_sdb.datasets import RFAM_PATH, RNA_SDB_PATH, load_rfamseq_fasta
from rna_sdb.utils import parse_alignment


def fetch_cm_for_family(rfam_id: str):
    """Fetch the covariance model for a given Rfam family"""
    # Path to the Rfam covariance model containing all families
    cmfile = RFAM_PATH / "cms" / "Rfam.cm.gz"
    outfile = RFAM_PATH / "cms" / f"{rfam_id}.cm"
    cmd = ["cmfetch", str(cmfile), rfam_id]
    with open(outfile, "w") as f:
        subprocess.run(cmd, check=True, stdout=f)


def align_seq_to_cm(
    rfam_id: str,
    cpu: int = mp.cpu_count(),
    fixtau: bool = False,
    mxsize: Optional[int] = None,
):
    """Align the Rfam sequences to the covariance model using 'cmalign'"""
    cmfile = RFAM_PATH / "cms" / f"{rfam_id}.cm"
    fasta = RFAM_PATH / "rfamseq_nonseed" / f"{rfam_id}.nonseed.fa.gz"
    seed_align = RFAM_PATH / "seed_alignments" / f"{rfam_id}.seed.sto"
    cmd = [
        "cmalign",
        "--noprob",
        "--cpu",
        str(cpu),
        "--mapali",
        str(seed_align),
        "--mapstr",
    ]
    if fixtau:
        # Do not adjust tau (tighten bands) until mx size is < limit
        # http://eddylab.org/infernal/Userguide.pdf
        cmd += ["--fixedtau"]
    if mxsize:
        # Set maximum allowable DP matrix size to <x> Mb
        # If not specified, defaults to 1024.0 MB
        cmd += ["--mxsize", str(mxsize)]
    cmd += [str(cmfile), str(fasta)]

    outfile = RFAM_PATH / "cmalign_output" / f"{rfam_id}.sto"
    outfile.parent.mkdir(exist_ok=True)

    # Check if there are any non-seed sequences
    with gzip.open(fasta, "rt") as f:
        try:
            next(f)
        except StopIteration:
            # There is no non-seed sequences to align.
            # Copy the seed_align file to the output of cmalign
            shutil.copy(seed_align, outfile)

            return

    with open(outfile, "w") as f:
        subprocess.run(cmd, check=True, stdout=f)


def parse_cmalign_output(
    align_path: Union[str, Path], rfam_id: str, filter_canonical: Optional[bool] = True
) -> pd.DataFrame:
    """Parse the Stockholm file output of 'cmalign'

    Check against the corresponding Rfamseq fasta to double check if the parsed
    sequence is correct.

    Args:
        align_path (Union[str, Path]): Path to the Stockholm file
        filter_canonical (Optional[bool], optional): Filter out non-canonical bases.
                                                    Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing sequences and their structures, obtained by
                    projecting the consensus structure onto the aligned sequences
    """
    df, rfam_id_parsed, _ = parse_alignment(align_path)
    df = df.rename(columns={"seq_name": "seq_id"})

    if rfam_id_parsed is not None:  # if rfam_id was included in the Stockholm file
        assert (
            rfam_id == rfam_id_parsed
        ), f"Rfam ID mismatch: {rfam_id} != {rfam_id_parsed} from {align_path}"

    # Compare the parsed sequences against the Rfamseq fasta
    df_rfam = load_rfamseq_fasta(rfam_id, convert_u_to_t=True)
    df_rfam = df_rfam.drop_duplicates()
    df_rfam["seq_id"] = df_rfam["seq_id"].str.split().str[0]
    df_test = df.merge(df_rfam, on="seq_id", how="left")
    assert (df["seq_id"] == df_test["seq_id"]).all(), "df and df_rfam does not match"
    assert (
        df_test["seq_x"].str.replace("U", "T") == df_test["seq_y"]
    ).all(), "seq from df and df_rfam does not match"

    # Cleanup df
    if filter_canonical:
        df = df[~df["seq"].str.contains(r"[^ATCGU]", regex=True)]
    df = df.rename(columns={"seq_name": "seq_id"})

    return df


def run(
    cpu: int = mp.cpu_count(),
    fixtau: bool = False,
    mxsize: Optional[int] = None,
    resume: bool = False,
):
    print("Processing Rfam sequences by aligning to CMs...")
    print(f"Using {cpu} CPUs")

    # Fetch all Rfam families in Rfamseq
    rfamseq_fa = (RFAM_PATH / "rfamseq").glob("*.fa.gz")
    rfam_families = sorted([f.name.replace(".fa.gz", "") for f in rfamseq_fa])

    # Process each Rfam family
    for family in tqdm(rfam_families):
        # Check if the alignment file already exists
        if resume and (RFAM_PATH / "cmalign_output" / f"{family}.sto").exists():
            continue
        fetch_cm_for_family(family)
        align_seq_to_cm(family, cpu=cpu, fixtau=fixtau, mxsize=mxsize)

    print("Parse the alignment files in Stockholm format...")

    for align_path in tqdm(sorted((RFAM_PATH / "cmalign_output").glob("*.sto"))):
        rfam_id = align_path.stem

        if (
            resume
            and (RNA_SDB_PATH / "parsed_alignments" / f"{rfam_id}.csv.gz").exists()
        ):
            continue

        df = parse_cmalign_output(align_path, rfam_id)

        df["rfam_family"] = rfam_id

        # Save the Rfam family sequences
        (RNA_SDB_PATH / "parsed_alignments").mkdir(parents=True, exist_ok=True)
        df.to_csv(
            RNA_SDB_PATH / "parsed_alignments" / f"{rfam_id}.csv.gz",
            index=False,
            compression="gzip",
        )

    # Merge all parsed alignments into a single file
    df_list = [
        pd.read_csv(f, compression="gzip")
        for f in sorted((RNA_SDB_PATH / "parsed_alignments").glob("*.csv.gz"))
    ]
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(
        RNA_SDB_PATH / "parsed_alignments" / "rfam_parsed_align.csv.gz",
        index=False,
        compression="gzip",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes Rfam sequences by aligning to covariance models (CMs)"
        " using 'cmalign' and parsing the alignment files to extract sequences and "
        "structures."
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=mp.cpu_count(),
        help="Number of CPUs to use for alignment",
    )
    parser.add_argument(
        "--fixtau",
        action="store_true",
        help="Adjust tau (tighten bands) until mx size is < limit",
    )
    parser.add_argument(
        "--mxsize",
        type=int,
        default=2048,  # 2GB should work for all Rfamseq
        help="Set maximum allowable DP matrix size to <x> Mb",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from the last processed family",
    )
    args = parser.parse_args()

    run(args.cpu, fixtau=args.fixtau, mxsize=args.mxsize, resume=args.resume)
