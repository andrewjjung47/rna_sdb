# Datasets used in RNA-SDB

This directory contains datasets used for processing RNA-SDB or benchmarking of RNA SS prediction methods.

List of datasets:

- bpRNA
- bpRNA-new
- ArchiveII
- RNAStralign
- Rfam

## Initial setup

Run the following command inside this directory to download and setup all of the datasets:

```bash
bash setup.sh
```

`setup.sh` will run an additional Python script, `rfam/setup_rfam.py`, to download Rfamseq and apply appropriate processing on the Rfam files.
