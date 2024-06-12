import torch

from rna_sdb.run_model import cli_main

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        print("Unable to activate TensorCore")

    cli_main()
