#!/bin/bash
set -euo pipefail

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_1 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_1.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_1 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_1.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_1 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_1.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_2 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_2.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_2 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_2.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_2 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_2.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_3 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_3.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_3 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_3.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_3 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_3.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_4 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_4.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_4 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_4.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_4 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_4.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_5 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_5.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_5 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_5.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_5 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_5.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_6 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_6.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_6 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_6.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_6 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_6.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_7 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_7.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_7 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_7.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_7 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_7.2.ckpt'

python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_8 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_8.0.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_8 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_8.1.ckpt'
python run_train_rnasdb.py --config=config_ufold.yml test --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_weighted --data.weighted_sampling=True --data.split_name=split_8 --ckpt='../../results/saved_models/rnasdb_ufold/weighted/ufold_rnasdb_weighted_split_8.2.ckpt'