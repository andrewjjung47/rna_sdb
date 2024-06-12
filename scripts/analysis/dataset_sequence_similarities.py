import logging
import multiprocessing as mp
from typing import Dict

import pandas as pd
from IPython.display import display

from rna_sdb.datasets import (
    load_archiveII,
    load_bprna,
    load_bprna_new,
    load_rnastralign,
)
from rna_sdb.datasets.rna_sdb import train_cluster, train_test_filter


def load_datasets() -> dict:
    datasets = dict()
    datasets["ArchiveII"] = load_archiveII()
    datasets["bpRNA"] = load_bprna()
    datasets["bpRNA-new"] = load_bprna_new()
    datasets["RNAStrAlign"] = load_rnastralign()

    return datasets


def sequence_similarity_analysis(
    name: str, dataset: pd.DataFrame, datasets: Dict[str, pd.DataFrame]
):
    logging.info(f"Running sequence similarity analysis for {name} dataset...")
    logging.info(f"Number of sequences in {name} dataset: {len(dataset)}")

    dataset_filtered = train_cluster(dataset, num_threads=mp.cpu_count())
    num_unique = len(dataset_filtered[dataset_filtered["seed"]])
    logging.info(f'Number of unique sequences in "{name}" dataset: ' f"{num_unique}")

    results = [(name, name, num_unique)]
    for other_name, other_dataset in datasets.items():
        if other_name == name:
            continue

        dataset_filtered = train_test_filter(
            dataset, other_dataset, num_threads=mp.cpu_count()
        )
        logging.info(
            f'Number of unique sequences in "{name}" dataset that are '
            f'not in "{other_name}" dataset: {len(dataset_filtered)}'
        )
        results.append((name, other_name, len(dataset_filtered)))

    return results


def run():
    datasets = load_datasets()

    logging.info("Number of structures in each dataset:")

    for name, dataset in datasets.items():
        logging.info(f"{name}: {len(dataset)}")

    # Run dataset sequence similarity analysis
    df_bprna = datasets["bpRNA"]
    df_rnastralign = datasets["RNAStrAlign"]
    analysis_datasets = {
        "ArchiveII": datasets["ArchiveII"],
        "bpRNA-new": datasets["bpRNA-new"],
        "TR0": df_bprna[df_bprna["split"] == "TR0"],
        "TS0": df_bprna[df_bprna["split"] == "TS0"],
        "RNAstralign-train": df_rnastralign[df_rnastralign["split"] == "train"],
        "RNAstralign-test": df_rnastralign[df_rnastralign["split"] == "test"],
    }

    results = []
    for name, dataset in analysis_datasets.items():
        results += sequence_similarity_analysis(name, dataset, analysis_datasets)

    logging.info("Summary of sequence similarity analysis:")

    df = pd.DataFrame(
        results, columns=["dataset", "compared_dataset", "num_unique_sequences"]
    )
    pd.set_option("display.max_rows", None)
    display(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    run()
