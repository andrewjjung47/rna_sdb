import json
from pathlib import Path

import mlcroissant as mlc


def run():
    # FileObjects and FileSets define the resources of the dataset.
    distribution = [
        # gpt-3 is hosted on a GitHub repository:
        mlc.FileObject(
            id="gd-repository",
            name="gd-repository",
            description="RNA-SDB repository on Google Drive.",
            content_url="https://drive.google.com/drive/folders/1fYIsnzLQEFDiwZd0IiA1LF-tzPr16fF9?usp=drive_link",
            encoding_format="git+https",
            sha256="main",
        ),
        # Within that repository, a FileSet lists all CSV files:
        mlc.FileSet(
            id="pq-files",
            name="pq-files",
            description="Parquet files for training and test splits that are hosted on the Google Drive.",
            contained_in=["gd-repository"],
            encoding_format="application/x-parquet",
            includes="split_*_cache_*.pq",
        ),
    ]
    record_sets = [
        # RecordSets contains records in the dataset.
        mlc.RecordSet(
            id="pq",
            name="pq",
            # Each record has one or many fields...
            fields=[
                # Fields can be extracted from the FileObjects/FileSets.
                mlc.Field(
                    id="pq/seq_id",
                    name="seq_id",
                    description="Rfam sequence ID",
                    data_types=mlc.DataType.TEXT,
                    source=mlc.Source(
                        file_set="pq-files",
                        # Extract the field from the column of a FileObject/FileSet:
                        extract=mlc.Extract(column="seq_id"),
                    ),
                ),
                mlc.Field(
                    id="pq/seq",
                    name="seq",
                    description="RNA sequence",
                    data_types=mlc.DataType.TEXT,
                    source=mlc.Source(
                        file_set="pq-files",
                        extract=mlc.Extract(column="seq"),
                    ),
                ),
                mlc.Field(
                    id="pq/db_structure",
                    name="db_structure",
                    description="RNA secondary structure, represented as in dot-bracket notation",
                    data_types=mlc.DataType.TEXT,
                    source=mlc.Source(
                        file_set="pq-files",
                        extract=mlc.Extract(column="db_structure"),
                    ),
                ),
                mlc.Field(
                    id="pq/rfam_family",
                    name="rfam_family",
                    description="RNA family",
                    data_types=mlc.DataType.TEXT,
                    source=mlc.Source(
                        file_set="pq-files",
                        extract=mlc.Extract(column="rfam_family"),
                    ),
                ),
                mlc.Field(
                    id="pq/cluster_id",
                    name="cluster_id",
                    description="Cluster ID",
                    data_types=mlc.DataType.INTEGER,
                    source=mlc.Source(
                        file_set="pq-files",
                        extract=mlc.Extract(column="cluster_id"),
                    ),
                ),
                mlc.Field(
                    id="pq/cluster_size",
                    name="cluster_size",
                    description="Cluster size",
                    data_types=mlc.DataType.INTEGER,
                    source=mlc.Source(
                        file_set="pq-files",
                        extract=mlc.Extract(column="cluster_size"),
                    ),
                ),
            ],
        )
    ]

    # Metadata contains information about the dataset.
    metadata = mlc.Metadata(
        name="RNA-SDB",
        # Descriptions can contain plain text or markdown.
        description=(
            "RNA-SDB is a large-scale RNA SS dataset that will improve training "
            "and benchmarking of deep learning models for RNA SS prediction. "
            "RNA-SDB consists of 3,100,307 structures from 4,168 RNA families, "
            "which has 200-fold more RNA structures and 1.5 times more RNA "
            "families than the largest existing dataset. Furthermore, RNA-SDB is"
            " designed with family-fold CV, in which training and test sets are "
            "split by families, to allow for a rigorous assessment of "
            "inter-family generalization."
        ),
        url="https://github.com/andrewjjung47/rna_sdb",
        distribution=distribution,
        record_sets=record_sets,
    )

    print(metadata.issues.report())

    croissant_path = Path(__file__).resolve().parents[2] / "rnasdb_croissant.json"
    with open(croissant_path, "w") as f:
        content = metadata.to_json()
        content = json.dumps(content, indent=2)
        print(content)
        f.write(content)
        f.write("\n")  # Terminate file with newline


if __name__ == "__main__":
    run()
