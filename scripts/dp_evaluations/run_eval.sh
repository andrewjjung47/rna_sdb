#!/bin/bash
set -euo pipefail

python evaluate_archiveII.py --num_processes=16 --chunk_size=32 --save
python evaluate_rnasdb.py --num_processes=112 --chunk_size=32 --save