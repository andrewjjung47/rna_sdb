#!/bin/bash
set -euo pipefail

# Subsequent commands will be executed from the directory containing this script
cd $(dirname $0)

# Setup bpRNA dataset from SPOT-RNA paper. Rather than the original version in 
# ST file format, download the version from MXfold2 paper in BPSEQ format
curl -OL https://github.com/mxfold/mxfold2/releases/download/v0.1.1/bpRNA.tar.gz
tar -xzf bpRNA.tar.gz
# Clean up the directory structure
mv data/bpRNA_dataset-canonicals bprna
rm bpRNA.tar.gz
rm -r data

# Setup ArchiveII dataset. Download the version processed from MXfold2 paper
curl -OL https://github.com/mxfold/mxfold2/releases/download/v0.1.1/archiveII.tar.gz
tar -xzf archiveII.tar.gz
# Clean up the directory structure
mv data/archiveII ./
rm archiveII.tar.gz
rm -r data

# Setup RNAStralign dataset. Download the version processed from MXfold2 paper
curl -OL https://github.com/mxfold/mxfold2/releases/download/v0.1.1/RNAStrAlign.tar.gz
tar -xzf RNAStrAlign.tar.gz
# Clean up the directory structure
mv data/RNAStrAlign ./rnastralign
rm RNAStrAlign.tar.gz
rm -r data

# Setup bpRNA-new dataset. Download the version processed from MXfold2 paper
curl -OL https://github.com/mxfold/mxfold2/releases/download/v0.1.0/bpRNAnew.tar.gz
tar -xzf bpRNAnew.tar.gz
# Clean up the directory structure
mv data/bpRNAnew_dataset/bpRNAnew.nr500.canonicals ./bprna_new
rm bpRNAnew.tar.gz
rm -r data

# Setup Rfam dataset. Download the necessary Rfam files
mkdir -p rfam/cms
curl https://ftp.ebi.ac.uk/pub/databases/Rfam/14.10/Rfam.cm.gz -o rfam/cms/Rfam.cm.gz
curl https://ftp.ebi.ac.uk/pub/databases/Rfam/14.10/Rfam.seed.gz -o rfam/Rfam.seed.gz
python rfam/setup_rfam.py --rfamseq --seed_align