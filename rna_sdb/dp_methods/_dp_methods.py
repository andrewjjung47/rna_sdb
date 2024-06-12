import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Literal, Union

import RNA
from tqdm import tqdm

# Arnie package uses arnie file to find paths to different methods supported
# Unfortunately, it only supports static paths in the arnie file. To avoid
# hardcoding paths in the arnie file, it will be dynamically generated from
# the template.

# First read the base arnie file
with open(Path(__file__).resolve().parent / "arnie_file_base.txt", "r") as f:
    arnie_file = f.read()

# Dynamically generate the arnie file with the paths to different methods
# This file is dynamically generated from the template
conda_path = Path(os.environ["CONDA_PREFIX"])
conda_bin_path = str(conda_path / "bin")
arnie_file = arnie_file.format(
    rnastructure=conda_bin_path,
    contrafold=conda_bin_path,
    linearfold=conda_bin_path,
)

with open(Path(__file__).resolve().parent / "arnie_file.txt", "w") as f:
    f.write(arnie_file)


# Add ARNIEFILE environment variable which contains paths to different methods
# supported in arnie
os.environ["ARNIEFILE"] = str(Path(__file__).resolve().parent / "arnie_file.txt")

# Add DATAPATH environment variable pointing to parameter files for RNAstructure package
os.environ["DATAPATH"] = str(conda_path / "share/rnastructure/data_tables")


from arnie.mfe import mfe  # noqa: E402


def _run_predictions(
    seqs: Union[str, List[str]],
    package: Literal["vienna_2", "contrafold_2", "rnastructure"],
    linear: bool = False,
    prog_bar: bool = False,
) -> Union[str, List[str]]:
    """Wrapper function to run Arnie's mfe predictions"""
    if isinstance(seqs, str):
        # If only one sequence is passed, return the prediction as a string
        # This can happen when running predictions with multiprocessing
        return mfe(seqs, package=package, linear=linear)

    preds = []
    for seq in tqdm(seqs, disable=not prog_bar):
        preds.append(mfe(seq, package=package, linear=linear))

    return preds


def _run_rnafold(
    seqs: Union[str, List[str]], prog_bar: bool = False
) -> Union[str, List[str]]:
    """Wrapper for running RNAfold

    Arnie can crash when running RNAfold in parallel processes, so use
    ViennaRNA's Python interface instead.
    https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/api_python.html
    """
    if isinstance(seqs, str):
        # If only one sequence is passed, return the prediction as a string
        # This can happen when running predictions with multiprocessing
        fc = RNA.fold_compound(seqs)
        ss, _ = fc.mfe()
        return ss

    preds = []
    for seq in tqdm(seqs, disable=not prog_bar):
        fc = RNA.fold_compound(seq)
        ss, _ = fc.mfe()
        preds.append(ss)

    return preds


def _run_rnastructure(
    seqs: Union[str, List[str]], prog_bar: bool = False
) -> Union[str, List[str]]:
    """Wrapper for running RNAstructure"""
    return _run_predictions(seqs, package="rnastructure", prog_bar=prog_bar)


def _run_contrafold(
    seqs: Union[str, List[str]], prog_bar: bool = False
) -> Union[str, List[str]]:
    """Wrapper for running CONTRAfold"""
    return _run_predictions(seqs, package="contrafold_2", prog_bar=prog_bar)


def _run_linearfold_v(
    seqs: Union[str, List[str]], prog_bar: bool = False
) -> Union[str, List[str]]:
    """Wrapper for running LinearFold with ViennaRNA parameters"""
    # TODO: With conda installed linearfold, get error
    return _run_predictions(seqs, package="vienna_2", linear=True, prog_bar=prog_bar)


def _run_linearfold_c(
    seqs: Union[str, List[str]], prog_bar: bool = False
) -> Union[str, List[str]]:
    """Wrapper for running LinearFold with CONTRAfold parameters"""
    # TODO: With conda installed linearfold, get error
    return _run_predictions(
        seqs, package="contrafold_2", linear=True, prog_bar=prog_bar
    )


def _batch_predictions(
    seqs: List[str],
    pred_func: callable,
    num_processes: int = 1,
    chunksize: int = 1,
    prog_bar: bool = False,
) -> List[str]:
    """Helper function to run predictions in parallel."""
    if num_processes == 1:
        pred_list = pred_func(seqs, prog_bar=prog_bar)
    else:
        # To make this work with Arnie initialization, use fork over spawn
        with mp.get_context("fork").Pool(num_processes) as pool:
            pred_list = list(
                tqdm(
                    pool.imap(pred_func, seqs, chunksize=chunksize),
                    total=len(seqs),
                    disable=not prog_bar,
                )
            )

    return pred_list


def run_rnafold(
    seqs: List[str], num_processes: int = 1, chunksize: int = 1, prog_bar: bool = False
) -> List[str]:
    return _batch_predictions(
        seqs, _run_rnafold, num_processes, chunksize, prog_bar=prog_bar
    )


def run_rnastructure(
    seqs: List[str], num_processes: int = 1, chunksize: int = 1, prog_bar: bool = False
) -> List[str]:
    return _batch_predictions(
        seqs, _run_rnastructure, num_processes, chunksize, prog_bar=prog_bar
    )


def run_contrafold(
    seqs: List[str], num_processes: int = 1, chunksize: int = 1, prog_bar: bool = False
) -> List[str]:
    return _batch_predictions(
        seqs, _run_contrafold, num_processes, chunksize, prog_bar=prog_bar
    )


def run_linearfold_v(
    seqs: List[str], num_processes: int = 1, chunksize: int = 1, prog_bar: bool = False
) -> List[str]:
    return _batch_predictions(
        seqs, _run_linearfold_v, num_processes, chunksize, prog_bar=prog_bar
    )


def run_linearfold_c(
    seqs: List[str], num_processes: int = 1, chunksize: int = 1, prog_bar: bool = False
) -> List[str]:
    return _batch_predictions(
        seqs, _run_linearfold_c, num_processes, chunksize, prog_bar=prog_bar
    )
