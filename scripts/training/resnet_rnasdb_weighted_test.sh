#!/bin/bash
set -euo pipefail

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_1 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_1.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_1 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_1.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_1 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_1.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_2 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_2.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_2 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_2.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_2 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_2.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_3 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_3.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_3 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_3.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_3 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_3.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_4 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_4.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_4 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_4.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_4 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_4.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_5 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_5.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_5 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_5.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_5 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_5.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_6 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_6.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_6 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_6.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_6 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_6.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_7 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_7.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_7 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_7.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_7 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_7.2.ckpt/model.ckpt'

python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_8 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_8.0.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_8 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_8.1.ckpt/model.ckpt'
python run_train_rnasdb.py --config=config_resnet.yml test --wandb_config=config_wandb.yml  --wandb_jobtype=rnasdb_weighted --data.split_name=split_8 --ckpt_path='../../results/saved_models/rnasdb_resnet/weighted/resnet_rnasdb_weighted_split_8.2.ckpt/model.ckpt'
