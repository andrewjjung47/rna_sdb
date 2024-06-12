import argparse
import gzip
import re
import urllib.request
from pathlib import Path
from typing import List

from tqdm import tqdm

from rna_sdb.datasets import RFAM_PATH
from rna_sdb.utils import parse_alignment

RFAM_VERSION = "14.10"


def setup_rfamseq():
    """Download Rfamseq fasta files from Rfam FTP server."""
    rfamseq_dir = RFAM_PATH / "rfamseq"
    rfamseq_dir.mkdir(exist_ok=True)

    print("Downloading files...")
    url_base = f"https://ftp.ebi.ac.uk/pub/databases/Rfam/{RFAM_VERSION}/fasta_files/"
    filenames = [f"RF{i:05}.fa.gz" for i in range(1, 4301)]

    for filename in tqdm(filenames):
        try:
            urllib.request.urlretrieve(url_base + filename, str(rfamseq_dir / filename))
        except urllib.error.HTTPError:
            print(
                f"Error downloading {filename}:"
                " most likely this does not exist in Rfam",
            )


def _flush_rfam_seed_family(rfam_seed_family: List[str], save_path_base: Path):
    """Write the seed alignment for a single family to a file."""
    assert rfam_seed_family[0].startswith("# STOCKHOLM 1.0")
    assert rfam_seed_family[1] == "\n"
    assert rfam_seed_family[2].startswith("#=GF AC")

    match = re.match(r"#=GF\s+AC\s+(RF\d+)", rfam_seed_family[2])
    rfam_family = match.group(1)

    with open(save_path_base / f"{rfam_family}.seed.sto", "w") as f_seed:
        f_seed.writelines(rfam_seed_family)


def setup_seed_alignments():
    """Split the concatenated seed alignments from Rfam.seed.gz into individual
    seed alignment files for each family.
    """
    rfam_seed = RFAM_PATH / "Rfam.seed.gz"
    save_path_base = RFAM_PATH / "seed_alignments"
    save_path_base.mkdir(exist_ok=True)

    with gzip.open(rfam_seed, "rt", encoding="latin-1") as f:
        rfam_seed_family = []  # List of lines for a single seed alignment
        for line in f:
            if line == "//\n":  # End of a seed alignment
                rfam_seed_family.append(
                    "//"
                )  # Stockholm format requires "//" at the end
                _flush_rfam_seed_family(rfam_seed_family, save_path_base)

                rfam_seed_family = []
            else:
                rfam_seed_family.append(line)


def _filter_nonseed_seq(rfamseq_file: Path, seed_seq_ids: set, output_file: Path):
    """Filter out non-seed sequences from Rfamseq file and write to a new fasta file."""
    with gzip.open(rfamseq_file, "rt") as f:
        with gzip.open(output_file, "wt") as f_out:
            for line in f:
                if not line.startswith(">"):
                    raise ValueError("Expected sequence id line first")

                seq_id = line.split()[0][1:]

                if seq_id in seed_seq_ids:
                    # Skip seed sequence
                    next(f)
                    continue

                # Write the non-seed sequence
                f_out.write(line)

                line = next(f)
                f_out.write(line)


def setup_rfamseq_nonseed():
    """Create non-seed Rfamseq files by filtering out seed sequences from Rfamseq."""
    print("Setting up non-seed Rfamseq...")

    rfamseq_dir = RFAM_PATH / "rfamseq"
    assert rfamseq_dir.exists(), "Rfamseq needs to be setup first"
    rfam_seed_base = RFAM_PATH / "seed_alignments"
    assert rfam_seed_base.exists(), "Seed alignments need to be setup first"

    nonseed_dir = RFAM_PATH / "rfamseq_nonseed"
    nonseed_dir.mkdir(exist_ok=True)

    for rfamseq_file in tqdm(rfamseq_dir.glob("*.fa.gz")):
        rfam_family = str(rfamseq_file.name).replace(".fa.gz", "")
        seed_align = rfam_seed_base / f"{rfam_family}.seed.sto"

        # Parse the seed alignment to get the seed sequence names
        df, rfam_family_parsed, _ = parse_alignment(seed_align, rfam_family)
        assert rfam_family == rfam_family_parsed

        rfamseq_file = rfamseq_dir / f"{rfam_family}.fa.gz"
        output_file = nonseed_dir / f"{rfam_family}.nonseed.fa.gz"
        _filter_nonseed_seq(rfamseq_file, set(df["seq_name"]), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to setup Rfam files")
    parser.add_argument(
        "--rfamseq", action="store_true", help="Setup Rfamseq fasta files"
    )
    parser.add_argument(
        "--seed_align",
        action="store_true",
        help="Setup seed alignments files for each family",
    )
    parser.add_argument(
        "--nonseed_seq",
        action="store_true",
        help="Setup non-seed Rfamseq",
    )
    args = parser.parse_args()

    if args.rfamseq:
        setup_rfamseq()

    if args.seed_align:
        setup_seed_alignments()

    if args.nonseed_seq:
        setup_rfamseq_nonseed()

    if not args.rfamseq and not args.seed_align and not args.nonseed_seq:
        print("No action specified. Use --help to see available options.")
