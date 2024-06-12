from pathlib import Path

import yaml
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from rna_sdb.utils import setup_output_dir


class RNASDBLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument("--checkpoint_prefix", default=None)
        parser.add_argument("--wandb_config", default=None)
        parser.add_argument("--wandb_jobtype", default=None)

    def before_instantiate_classes(self) -> None:
        checkpoint_prefix = self._get(self.config, "checkpoint_prefix")
        checkpoint_dir = setup_output_dir(output_prefix=checkpoint_prefix)

        print(f"Checkpointing to: {checkpoint_dir}")

        trainer_config = self._get(self.config, "trainer")
        trainer_config["default_root_dir"] = checkpoint_dir

        for logger in trainer_config["logger"]:
            if logger["class_path"] == "lightning.pytorch.loggers.WandbLogger":
                logger["init_args"]["save_dir"] = checkpoint_dir
                logger["init_args"]["job_type"] = self._get(
                    self.config, "wandb_jobtype"
                )

                # If there is a wandb_config file, load it
                wandb_config_file = self._get(self.config, "wandb_config")
                if wandb_config_file is not None:
                    with open(Path(__file__).parent / wandb_config_file, "r") as f:
                        wandb_config_yml = yaml.safe_load(f)

                    logger["init_args"]["entity"] = wandb_config_yml["entity"]
                    logger["init_args"]["project"] = wandb_config_yml["project"]

                # Save select run configs
                wandb_config = logger["init_args"]["config"]
                if wandb_config is None:
                    wandb_config = {}

                # From data config
                data_config = self._get(self.config, "data")["init_args"]
                wandb_config["split_name"] = data_config["split_name"]

                # From other CLI config
                wandb_config["seed"] = self._get(self.config, "seed_everything")

                logger["init_args"]["config"] = wandb_config


def cli_main(args: ArgsType = None):
    RNASDBLightningCLI(
        args=args,
    )


if __name__ == "__main__":
    cli_main()
