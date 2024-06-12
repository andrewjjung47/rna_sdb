from typing import Literal

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pydantic_cli import run_and_exit
from ribonanza.datasets.synthetic import SyntheticDataModule
from ribonanza.models import ModelWrapper, RecurrentTransformer, VanillaTransformer
from utils import BaseTrainConfig, setup_checkpoint_directories, setup_wandb_logger


class TrainResNetConfig(BaseTrainConfig):
    label: Literal["rnafold", "linearfold", "linearfold_contra", "linearpartition"]

    # ===========
    # Model Fields
    # ===========
    dim: int = 64
    num_heads: int = 8
    num_groups: int = 8
    kernel_size: int = 17

    # ===========
    # Dataset Fields
    # ===========
    batch_size: int = 64
    num_workers: int = 4
    train_subsample: float = None

    # ===============
    # Training Fields
    # ===============
    max_epochs: int = 100
    lr: float = 1e-4

    # ==============
    # Logging Fields
    # ==============
    wandb: bool = True
    wandb_entity: str = "deep-genomics-ml"
    wandb_project: str = "andrew_sandbox"
    wandb_group_prefix: str = "atom1-synthetic-training/"
    wandb_run_name: str = None
    wandb_job_type: str = "synthetic-training"

    # ===========
    # Pre-generated synthetic datasets
    # ===========
    wandb_artifact_train: str = (
        "deep-genomics-ml/ribonanza/ribonanza-synthetic-dataset-training:v3"
    )
    wandb_artifact_test: str = (
        "deep-genomics-ml/ribonanza/ribonanza-synthetic-dataset-test:v2"
    )
    wandb_artifact_test_longer: str = (
        "deep-genomics-ml/ribonanza/ribonanza-synthetic-dataset-longer:v4"
    )


def main(cfg: TrainResNetConfig):
    cfg.checkpoint_dir = setup_checkpoint_directories(cfg.checkpoint_dir)
    print(f"Running experiment with config: {cfg}")

    logger = setup_wandb_logger(cfg)

    dataset = SyntheticDataModule(
        labels=[cfg.label],
        cfg=dict(cfg),
        batch_size=cfg.batch_size,
    )

    optimizer_init = lambda params: torch.optim.Adam(params, lr=cfg.lr)
    # Initialize and load model
    # We use dict(cfg) so checkpointing does not rely on TrainRibonanzaConfig
    if cfg.model_type == "recurrent-tf":
        model_base = RecurrentTransformer(
            cfg.dim,
            cfg.kernel_size,
            num_heads=cfg.num_heads,
            num_groups=cfg.num_groups,
        )
        model = ModelWrapper(model_base, optimizer_init)
    else:
        raise NotImplementedError(f"Unknown model type: {cfg.model_type}")

    # Setup checkpoints
    callbacks = [
        EarlyStopping(monitor="val/loss", mode="min", patience=10, verbose=True),
        ModelSummary(max_depth=3),
    ]
    if cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                save_top_k=3,
                monitor="val/loss",
                mode="min",
                save_last=True,
                verbose=True,
            )
        )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.num_gpus,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        fast_dev_run=cfg.debug,
        enable_progress_bar=cfg.progress_bar,
        callbacks=callbacks,
        logger=logger,
    )

    # Run training
    trainer.fit(
        model=model,
        datamodule=dataset,
        # ckpt_path=cfg.checkpoint_dir,  # checkpoint to resume training TODO: fix this
    )

    # Run testing
    trainer.test(model=model, datamodule=dataset)

    wandb.finish()  # Close Wandb process for a new run


if __name__ == "__main__":
    run_and_exit(TrainResNetConfig, main)
