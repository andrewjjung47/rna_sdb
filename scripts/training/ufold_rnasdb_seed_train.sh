#!/bin/bash
set -euo pipefail

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_1 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_1 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_1 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_2 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_2 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_2 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_3 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_3 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_3 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_4 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_4 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_4 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_5 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_5 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_5 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_6 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_6 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_6 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_7 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_7 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_7 --seed_everything=2

python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_8 --seed_everything=0
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_8 --seed_everything=1
python run_train_rnasdb.py --config=config_ufold.yml fit --wandb_config=config_wandb.yml --wandb_jobtype=rnasdb_seedonly --data.weighted_sampling=False --data.seed_only=True --trainer.limit_val_batches=1.0 --data.split_name=split_8 --seed_everything=2