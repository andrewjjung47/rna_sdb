import tempfile
import unittest

from rna_sdb.dataset import load_bprna
from rna_sdb.models.mxfold2 import train


class TestMXfold2Train(unittest.TestCase):
    def test_train(self):
        dataset = load_bprna()
        dataset = dataset[dataset["split"] == "VL0"].head(20)

        with tempfile.NamedTemporaryFile() as out_param, tempfile.NamedTemporaryFile() as out_config:  # noqa
            train(
                train_dataset=dataset,
                model_type="MixC",
                output_param=out_param.name,
                output_config=out_config.name,
                gpu_id=0,
                conda_env="mxfold2",
            )


if __name__ == "__main__":
    unittest.main()
