# RNA-SDB: RNA Secondary Structure Database

## Overview

RNA-SDB is a large-scale RNA SS dataset that can improve training and benchmarking of deep learning models for RNA SS prediction. RNA-SDB consists of 3,100,307 structures from 4,168 RNA families, which has 200-fold more RNA structures and 1.5 times more RNA families than the largest existing dataset. Furthermore, we designed RNA-SDB with family-fold CV, in which training and test sets are split by families, to allow for a rigorous assessment of inter-family generalization.

## How to get started

### Initial setup

#### Initializing Conda environment and installing `rna_sdb` package

Create a Conda environment by using `environment.yml`:

```bash
conda env install -f environment.yml
```

Then, install `rna_sdb` package:

```bash
pip install .
```

#### Datasets setup

`bash datasets/setup.sh` will download dataset files and process them. Please refer to `datasets/README.md` for more information.

### Using pre-built splits

To facilitate easy training and benchmarking for a broader audience of researcher, we provide eight pre-built training and test splits that can be used with ArchiveII. Each of the split uses one of the eight families from ArchiveII as part of the test.

RNA-SDB dataset is defined by `rna_sdb.datasets.RNASDB` class. For most use cases, the eight pre-built splits can be initialized by the class method `RNASDB.initialize_presplit`. You would need to download `split_*_cache_{train|test}.pq` from [Google Drive](https://drive.google.com/drive/folders/1fYIsnzLQEFDiwZd0IiA1LF-tzPr16fF9?usp=sharing) and save them into `datasets/rna_sdb` directory. After the review phase, we will be hosting them to Huggingface so that it will be more accessible to the public.

Example of how `RNASDB` is used can be found in `rna_sdb.data.RNASDBDatasetModule`. We use PyTorch Lightning for training and test scripts, and `RNASDBDatasetModule` is a Lightning DatasetModule for training on RNA-SDB. Here, `RNASDBDatasetModule` is initialized in `RNASDBDatasetModule.setup` function:

```python
def setup(self, stage: str):
    # Common setup for RNASDB
    split_train_val = True if stage == "fit" else False
    self.rnasdb = RNASDB.initialize_presplit(
        self.split_name, seed_only=self.seed_only, split_train_val=split_train_val
    )
    ...
```

This will initialize training, validation, and test splits which can be accessed by `df_train`, `df_val`, and `df_test`:

```python
def setup(self, stage: str):
    # Common setup for RNASDB
    ...

    if stage == "fit":
        self.dataset_train = self.dataset_class(self.rnasdb.df_train)
        self.dataset_val = self.dataset_class(self.rnasdb.df_val)
    elif stage == "test":
        self.dataset_test = self.dataset_class(self.rnasdb.df_test)
        ...
```

### Creating custom RNA-SDB splits

First, Rfamseq (or any other additional sequences) needs to be aligned and the sequence alignments need to be processed to RNA secondary structures. `scripts/dataset/process_rfam.py` can be used as is or adapted for new dataset.

Then, RNA-SDB needs to be split into training and test splits. This can be done by adapting `python scripts/dataset/split_rna_sdb.py`. Here, the key part is specifying which families will be included for testing

Finally, training split must undergo filtering to remove any examples with overlapping sequences to as those in test split. `python scripts/dataset/rna_sdb_filter.py` can be adapted.

## Reproducing the paper

### Dataset processing

#### Processing RNA secondary structures

`scripts/dataset/process_rfam.py` aligns Rfamseq and processes the alignments for RNA secondary structures. Run with the default parameters to obtain
the same results as the paper: `python scripts/dataset/process_rfam.py`

#### Processing the eight pre-built training and test splits

`scripts/dataset/split_rna_sdb.py` splits RNA-SDB into the eight training and test splits. Then, `scripts/dataset/rna_sdb_filter.py` filters training splits to ensure no structures have similar sequences to those in test splits.

Run with the default parameters to obtain the same results as the paper:

```bash
python scripts/dataset/split_rna_sdb.py --random_state=0
python scripts/dataset/rna_sdb_filter.py
```

### Benchmarking experiments

#### RNA secondary structure prediction methods

`rna_sdb.dp_methods` define wrappers for running and evaluating predictions for the three DP-methods in the paper (`RNAfold`, `RNAstructure`, and `CONTRAfold`). 

`rna_sdb.models` define reimplementation of the ResNet and Ufold models.

Modification of MXfold2 can be found in Git Submodule, `external/mxfold2`.

#### Evaluations

##### DP-methods

`scripts/dp_evaluations/evaluate_archiveII.py` and `scripts/dp_evaluations/evaluate_rnasdb.py` are Python scripts for evaluating specified DP-method on the eight pre-built RNA-SDB or ArchiveII test splits.

Run `bash scripts/dp_evaluations/run_eval.sh` to reproduce the benchmarking experiments on RNA-SDB and ArchiveII test splits (Table 4 and 5).

##### Deep learning models

`scripts/training/run_train_rnasdb.py` is the base Python script, which uses Lightning CLI, for training and evaluating the ResNet and Ufold models.

The following shell scripts use `run_train_rnasdb.py` to train and evaluate the ResNet model

- `scripts/training/resnet_rnasdb_weighted.sh` and `scripts/training/resnet_rnasdb_weighted_eval.sh` for training the ResNet model on RNA-SDB train splits and evaluating on RNA-SDB and ArchiveII test splits (Table 4 and 5)

The following shell scripts use `run_train_rnasdb.py` to train and evaluate Ufold

- `scripts/training/ufold_rnasdb_weighted_train.sh` and `scripts/training/ufold_rnasdb_weighted_test.sh` for training Ufold on RNA-SDB train splits and evaluating on RNA-SDB and ArchiveII test splits (Table 4 and 5)
- `scripts/training/ufold_archiveII_train.sh` and `scripts/training/ufold_archiveII_test.sh` for training Ufold on ArchiveII train splits and evaluating on ArchiveII test splits (Table 6)
- `scripts/training/ufold_rnasdb_non_weighted_train.sh` and `scripts/training/ufold_rnasdb_non_weighted_test.sh` for training Ufold without weighted sampling on RNA-SDB train splits and evaluating on RNA-SDB and ArchiveII test splits (Table 7)
- `scripts/training/ufold_rnasdb_seed_train.sh` and `scripts/training/ufold_rnasdb_seed_test.sh` for training Ufold with only seed sequences on RNA-SDB train splits and evaluating on RNA-SDB and ArchiveII test splits (Table 7)
