#!/bin/bash
set -euo pipefail

python create_weighted_datasets.py --split_name='split_1' --seed=0 --subsample=0.025
python create_weighted_datasets.py --split_name='split_2' --seed=0 --subsample=0.1
python create_weighted_datasets.py --split_name='split_3' --seed=0 --subsample=0.1
python create_weighted_datasets.py --split_name='split_4' --seed=0 --subsample=0.1
python create_weighted_datasets.py --split_name='split_5' --seed=0 --subsample=0.1
python create_weighted_datasets.py --split_name='split_6' --seed=0 --subsample=0.1
python create_weighted_datasets.py --split_name='split_7' --seed=0 --subsample=0.1
python create_weighted_datasets.py --split_name='split_8' --seed=0 --subsample=0.1