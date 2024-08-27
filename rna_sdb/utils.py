import gzip
import re
import string
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parents[1] / "results"

LEFT_BRACKETS = ["(", "<", "[", "{"]
RIGHT_BRACKETS = [")", ">", "]", "}"]
NON_PAIRING_CHARS = re.compile(r"[a-zA-Z_\,-\.\:~]")


# ==================================================================================================================
# File IO
# ==================================================================================================================


def read_fasta(
    file_path: Union[str, Path],
    filter_non_canonical: bool = False,
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
) -> List[Tuple[str, str]]:
    """Reads a fasta file and returns the sequence ids and sequences.

    Args:
        file_path (Union[str, Path]): Path to the fasta file.
        filter_non_canonical (bool, optional): Filter out non-canonical bases.
        Canonical bases are 'A', 'C', 'G', 'U', 'T'. Defaults to False.
        convert_to_upper (bool, optional): Convert sequences to uppercase.
        Defaults to True.
        convert_u_to_t (bool, optional): Convert 'U' to 'T'. Defaults to False.

    Returns:
        List[Tuple[str, str]]: List of tuples of sequence ids and sequences.
    """
    seqs = []

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {str(file_path)}")

    if file_path.suffix == ".gz":
        open_fn = gzip.open
    else:
        open_fn = open

    with open_fn(file_path, "rt") as file:
        for line in file:
            if not line.startswith(">"):  # sequence id line starts with >
                raise ValueError("Expected sequence id line first")

            seq_id = line[1:].strip()

            line = next(file).strip()  # sequence line
            if line.startswith(">"):
                raise ValueError("Expected sequence line after sequence id")

            seq = line

            if convert_to_upper:
                seq = seq.upper()

            # Skip sequences with non-canonical bases
            if filter_non_canonical and re.search(r"[^ACGTUacgtu]", seq):
                continue

            if convert_u_to_t:
                seq = seq.replace("U", "T")

            seqs.append((seq_id, seq))

    return seqs


def write_fasta(
    seqs: List[str],
    fasta_path: Union[str, Path],
    seq_ids: Optional[List[str]] = None,
    seq_id_prefix: Optional[str] = None,
):
    """Write a fasta file from list of sequences

    Optionally 'seq_ids' or 'seq_id_prefix' can be provided. Otherwise index
    in the list is used for the sequence id (i.e. 0, 1, ...).

    Args:
        seqs (List[str]): List of sequences
        fasta_path (Union[str, Path]): Path to the output fasta file
        seq_ids (Optional[List[str]], optional): List of sequence ids. Defaults to None.
        seq_id_prefix (Optional[str]): Prefix for sequence id.
    """
    if seq_ids is None:
        if seq_id_prefix is None:
            seq_id_prefix = ""  # no prefix
        else:
            seq_id_prefix = f"{seq_id_prefix}_"

        seq_ids = map(lambda idx: f"{seq_id_prefix}{idx}", range(len(seqs)))
    else:
        if seq_id_prefix is not None:
            raise ValueError("seq_id_prefix should be None when seq_ids is provided")

    with open(fasta_path, "w") as f:
        for seq_id, seq in zip(seq_ids, seqs):
            f.write(f">{seq_id}\n")
            f.write(f"{seq}\n")


def read_bpseq(
    file_path: Union[str, Path],
    convert_to_upper: bool = True,
    convert_u_to_t: bool = False,
) -> Tuple[str, Tuple[List[int], List[int]]]:
    """Read a BPseq format file and parse the sequence and base pair indices

    Args:
        file_path (Union[str, Path]): Path to the BPSEQ file

    Returns:
        str: sequence
        Tuple[List[int], List[int]]: base-pair indices, represented
            as two lists of 0-based indices of the first and second bases in base pairs
    """
    seq, pair_indices = "", ([], [])
    with open(file_path, "r") as f:
        for line in f:
            idx, base, pair_idx = line.split()
            idx, pair_idx = int(idx), int(pair_idx)
            seq += base

            if pair_idx != 0:  # 0 means no pair
                # 1-indexed to 0-indexed
                pair_indices[0].append(idx - 1)
                pair_indices[1].append(pair_idx - 1)

    if convert_to_upper:
        seq = seq.upper()

    if convert_u_to_t:
        seq = seq.replace("U", "T")

    return seq, pair_indices


def wuss_to_db(structure: str):
    # Non base-pairing characters
    regex = re.compile(r"[a-zA-Z_\,-\.\:~]")

    db_st = []
    for i in structure:
        if i in LEFT_BRACKETS:
            db_st.append("(")
        elif i in RIGHT_BRACKETS:
            db_st.append(")")
        elif re.match(regex, i):
            db_st.append(".")

    db_st = "".join(db_st)

    assert len(db_st) == len(structure)

    return db_st


def parse_alignment(
    file_path: str, convert_to_upper: bool = True
) -> Tuple[pd.DataFrame, str, str]:
    with open(file_path, "r") as f:
        seqs = defaultdict(lambda: ["", []])  # save {seq_name: (seq, gap_indices)}
        ss_cons = ""
        rfam_id = (
            None  # it is possible that the alignment file does not contain Rfam ID
        )

        block_seqs = defaultdict(lambda: ["", []])  # seqs for each block
        for line in f:
            # Rfam ID line
            if match := re.match(r"#=GF\s+AC\s+(RF\d{5})", line):
                rfam_id = match.group(1)
            # SS_cons line
            elif match := re.match(r"#=GC\s+SS_cons\s+(\S+)", line):
                ss_cons += match.group(1)

                # Update block_seqs to the main seqs
                for seq_name, (seq, gapped_flags) in block_seqs.items():
                    seqs[seq_name][0] += seq
                    seqs[seq_name][1] += gapped_flags
                block_seqs = defaultdict(lambda: ["", []])  # reset for next block
            elif line.startswith("#"):  # skip rest of comments
                pass
            # Parse sequence lines
            elif match := re.match(r"(\S+)\s+(\S+)", line):
                seq_name, seq = match.groups()

                seq_name = re.sub(
                    r"\d+\|", "", seq_name
                )  # remove numbering added by cmalign if present

                if convert_to_upper:
                    seq = seq.upper()

                # Save gap indices and remove them
                gapped_flags = []  # 0 for gap, 1 for non-gap
                seq_processed = ""
                for c in seq:
                    if c in [".", "-", "_", "~"]:
                        gapped_flags.append(0)
                    elif c in string.ascii_uppercase:
                        gapped_flags.append(1)
                        seq_processed += c
                    else:
                        raise ValueError(f"Invalid character in sequence: {c}")

                assert sum(gapped_flags) == len(seq_processed)

                if seq_name in block_seqs:
                    assert seq_processed == block_seqs[seq_name][0]
                    assert gapped_flags == block_seqs[seq_name][1]

                    continue

                block_seqs[seq_name][0] = seq_processed
                block_seqs[seq_name][1] = gapped_flags

        if ss_cons is None:
            raise ValueError("Alignment file does not contain SS_cons")
        _ss_cons_sanity_check(ss_cons)

        df_data = []
        ss_cons_pair_indices = db2pairs(ss_cons)

        for seq_name, (seq, gapped_flags) in seqs.items():
            assert len(ss_cons) == len(gapped_flags)

            structure = _parse_projected_structure(
                ss_cons, ss_cons_pair_indices, gapped_flags
            )
            db_structure = wuss_to_db(structure)

            assert len(seq) == len(structure)
            assert db_structure.count("(") == db_structure.count(")")

            df_data.append((seq_name, seq, structure, db_structure))

    df = pd.DataFrame(df_data, columns=["seq_name", "seq", "structure", "db_structure"])

    return df, rfam_id, ss_cons


def _parse_projected_structure(
    ss_cons: str, pair_indices: List[Tuple[int, int]], gapped_flags: List[int]
):
    """Parse the projected structure from the SS_cons line and gap indices

    For the most part, characters in the SS_cons that are not corresponding to
    a gap is included in the projected structure.

    When a base in a base-pair (i.e. a bracket) is gapped, it is important that
    the corresponding base forming the pair (i.e. matched bracket) is also
    removed from the projected structure.
    """
    # First, check which of the base-pairs are gapped
    gap_indices = np.where(np.array(gapped_flags) == 0)[0]

    # Process gapped_pairs which consists of indices of gapped pairs
    pair_indices_zipped = list(zip(*pair_indices))
    if len(pair_indices_zipped) == 0:  # no base-pairing
        pair_indices_zipped = [[], []]

    gapped_pair_indices_1 = np.where(np.in1d(pair_indices_zipped[0], gap_indices))[
        0
    ]  # 1st base in pair is gapped
    gapped_pair_indices_2 = np.where(np.in1d(pair_indices_zipped[1], gap_indices))[
        0
    ]  # 2nd base in pair is gapped

    gapped_pair_indices = np.union1d(
        gapped_pair_indices_1, gapped_pair_indices_2
    )  # pairs where at least one base is gapped
    gapped_pairs = np.array(pair_indices)[
        gapped_pair_indices
    ].flatten()  # indices of bases forming pairs that are gapped

    # Then, construct the projected structure
    ss_cons_array = np.array([*ss_cons])  # convert to numpy array for indexing
    nongap_indices = np.nonzero(gapped_flags)[0]  # indices of non-gapped bases
    unmatched_indices = np.intersect1d(nongap_indices, gapped_pairs).astype(
        np.int64
    )  # indices of bases that are not gapped but their pairs are gapped
    ss_cons_array[unmatched_indices] = "."  # these bases no longer form pairs
    structure = "".join(ss_cons_array[nongap_indices])

    return structure


def _ss_cons_sanity_check(ss_cons: str):
    left_count, right_count = 0, 0
    for lb in LEFT_BRACKETS:
        left_count += ss_cons.count(lb)
    for rb in RIGHT_BRACKETS:
        right_count += ss_cons.count(rb)

    if left_count != right_count:
        raise ValueError("Unbalanced brackets in SS_cons")


# TODO: migrated from Alice's code base. Double check if this is correct
class AllPairs(object):
    def __init__(self, db_str):
        self.db_str = db_str
        self.pairs = (
            []
        )  # list of tuples, where each tuple is one paired positions (i, j)
        # hold on to all bracket groups
        self.bracket_round = PairedBrackets(left_str="(", right_str=")")
        self.bracket_square = PairedBrackets(left_str="[", right_str="]")
        self.bracket_triang = PairedBrackets(left_str="<", right_str=">")
        self.bracket_curly = PairedBrackets(left_str="{", right_str="}")

    def parse_db(self):
        # parse dot-bracket notation
        for i, s in enumerate(self.db_str):
            # add s into bracket collection, if paired
            # also check if any bracket group is completed, if so, flush
            if re.match(NON_PAIRING_CHARS, s):
                continue
            elif self.bracket_round.is_compatible(s):
                self.bracket_round.add_s(s, i)
                # if self.bracket_round.is_complete():
                #     self.pairs.extend(self.bracket_round.flush())
            elif self.bracket_square.is_compatible(s):
                self.bracket_square.add_s(s, i)
                # if self.bracket_square.is_complete():
                #     self.pairs.extend(self.bracket_square.flush())
            elif self.bracket_triang.is_compatible(s):
                self.bracket_triang.add_s(s, i)
                # if self.bracket_triang.is_complete():
                #     self.pairs.extend(self.bracket_triang.flush())
            elif self.bracket_curly.is_compatible(s):
                self.bracket_curly.add_s(s, i)
                # if self.bracket_curly.is_complete():
                #     self.pairs.extend(self.bracket_curly.flush())
            else:
                raise ValueError(
                    "Unrecognized character {} at position {}".format(s, i)
                )

        # check that all groups are empty!!
        bracket_groups = [
            self.bracket_round,
            self.bracket_curly,
            self.bracket_triang,
            self.bracket_square,
        ]
        for bracket in bracket_groups:
            if not bracket.is_empty():
                raise ValueError(
                    "Bracket group {}-{} not symmetric: left stack".format(
                        bracket.left_str, bracket.right_str, bracket.left_stack
                    )
                )

        # collect and sort all pairs
        pairs = []
        for bracket in bracket_groups:
            pairs.extend(bracket.pairs)
        pairs = sorted(pairs)
        self.pairs = pairs


class PairedBrackets(object):
    def __init__(self, left_str, right_str):
        self.left_str = left_str
        self.right_str = right_str
        self.pairs = []  # list of tuples (i, j)
        self.left_stack = []  # left positions

    def is_empty(self):
        return len(self.left_stack) == 0

    def is_compatible(self, s):
        return s in [self.left_str, self.right_str]

    def add_s(self, s, pos):
        if s == self.left_str:
            self.left_stack.append(pos)
        elif s == self.right_str:
            # pop on item from left_stack
            i = self.left_stack.pop()
            self.pairs.append((i, pos))
        else:
            raise ValueError(
                "Expect {} or {} but got {}".format(self.left_str, self.right_str, s)
            )


def db2pairs(s):
    ap = AllPairs(s)
    ap.parse_db()
    return ap.pairs


# ==================================================================================================================
# Miscellaneous
# ==================================================================================================================


def setup_output_dir(
    output_dir: Optional[Union[Path, str]] = None, output_prefix: Optional[str] = None
) -> Path:
    """Setup the output directory for saving files (e.g. checkpoints, logs, etc...)

    If output_dir is not specified, a random directory is created under OUTPUT_DIR.

    Args:
        output_dir (Optional[Union[Path, str]], optional): Output directory.
                                                        Defaults to None.
        output_prefix (Optional[str], optional): Prefix for the output directory.

    Returns:
        Path: Output directory that is initialized
    """

    if output_dir is None:  # if output_dir is not specified, create a random one
        if output_prefix is not None:
            output_dir = OUTPUT_DIR / output_prefix / str(uuid.uuid4())
        else:
            output_dir = OUTPUT_DIR / str(uuid.uuid4())
        assert not output_dir.exists()

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
