import math
import re
from typing import Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rna_sdb.datasets import RNASDB, load_archiveII, load_bprna

# ==================================================================================================================
# Dataset class and helper functions for generic models
# ==================================================================================================================

BASE_TO_ONE_HOT_LABEL = np.zeros([90, 4], dtype=np.float32)
BASE_TO_ONE_HOT_LABEL[[ord(base) for base in "ACGT"]] = np.eye(4)
BASE_TO_ONE_HOT_LABEL[ord("U")] = BASE_TO_ONE_HOT_LABEL[ord("T")]


def fast_one_hot(seq):
    bseq = seq.encode("ascii")
    encoding = BASE_TO_ONE_HOT_LABEL[memoryview(bseq)]
    return torch.tensor(encoding, dtype=torch.float)


def idx2matrix(idx: tuple, seq_len: int):
    pair_matrix = np.zeros((seq_len, seq_len), dtype=np.int8)

    if len(idx) == 0 or len(idx[0]) == 0:  # for case with no base-pairing
        return pair_matrix

    pair_matrix[idx[0], idx[1]] = 1
    # Make the matrix symmetric in case idx is not already
    pair_matrix[idx[1], idx[0]] = 1

    assert np.all(pair_matrix == pair_matrix.T)

    return pair_matrix


def collate_rna_data(data_list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch RNA sequence and structure data with different lengths.
    Returns padding mask indicating which elements are padded or not.
    """
    lengths = map(lambda data: data[0].shape[0], data_list)
    max_len = max(lengths)

    padded_x, padded_y = [], []
    for x, y in data_list:
        padded_x.append(
            F.pad(x, [0, 0, 0, max_len - x.shape[0]], mode="constant", value=0)
        )
        padded_y.append(
            F.pad(
                y,
                [0, max_len - y.shape[1], 0, max_len - y.shape[0]],
                mode="constant",
                value=0,
            )
        )

    return torch.stack(padded_x, dim=0), torch.stack(padded_y, dim=0)


class SecondaryStructure(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_col: str = "seq",
        pair_col: str = "pair_indices",
    ):
        super().__init__()

        self.seq_col, self.pair_col = seq_col, pair_col

        assert self.seq_col in df.columns and self.pair_col in df.columns
        self.df = df  # Keep the original DataFrame for debugging purposes
        self._data = df[[self.seq_col, self.pair_col]]  # Only keep the relevant columns

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        sequence, one_idx = row[self.seq_col], row[self.pair_col]

        # FIXME replacing non-canonical bases with N
        # since data.py does not support them yet! (some character go out of bound, e.g. 103)
        sequence = sequence.upper()
        sequence = re.sub(r"([^ACGTU])", "N", sequence)

        x = fast_one_hot(sequence)
        y = idx2matrix(one_idx, len(sequence))
        y = torch.Tensor(y)

        return x, y


# ==================================================================================================================
# Dataset class and helper functions for UFold
# ==================================================================================================================


def collate_with_mask(data_dict):
    """
    Batch RNA sequence and structure data with different lengths.
    Returns padding mask indicating which elements are padded or not.

    Most of the function is similar to rna_sdb.data.collate_rna_data, but added
    support for mask (in source dataset)
    change length to data[0].shape[1]   (current dataset x.shape: (17, L, L))
    updated padding
    """
    lengths = map(lambda data: data["x"].shape[1], data_dict)
    max_len = max(lengths)

    padded_x, padded_y, padded_m = [], [], []
    seqs = []
    for data in data_dict:
        x, y, m = data["x"], data["y"], data["m"]
        seq = data["seq"]

        # x: (ch, L, L)
        # y: (L, L)
        # m: (L, L)

        # torch padding described starting from the last dimension
        padded_x.append(
            F.pad(
                x,
                [
                    0,
                    max_len - x.shape[2],
                    0,
                    max_len - x.shape[1],
                    0,
                    0,  # no padding along ch dim
                ],
                mode="constant",
                value=0,
            )
        )
        padded_y.append(
            F.pad(
                y,
                [0, max_len - y.shape[1], 0, max_len - y.shape[0]],
                mode="constant",
                value=0,
            )
        )
        padded_m.append(
            F.pad(
                m,
                [0, max_len - m.shape[1], 0, max_len - m.shape[0]],
                mode="constant",
                value=0,
            )
        )
        seqs.append(seq)

    x, y, m = (
        torch.stack(padded_x, dim=0),
        torch.stack(padded_y, dim=0),
        torch.stack(padded_m, dim=0),
    )

    return {
        "seq": seqs,
        "x": x,
        "y": y,
        "m": m,
    }


class PairFeature:
    """the 17th feature, see Ufold paper supplementary note for explanation"""

    def __init__(self) -> None:
        pass

    def compute_feature(self, seq):
        # copied from https://github.com/uci-cbcl/UFold/blob/3c92fa184ae66e385214f3e4c1da6cf9bfd667f5/ufold/utils.py#L101 # noqa E501
        mat = np.zeros([len(seq), len(seq)])
        for i in range(len(seq)):
            for j in range(len(seq)):
                coefficient = 0
                for add in range(30):
                    if i - add >= 0 and j + add < len(seq):
                        score = self.paired(seq[i - add], seq[j + add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * self.Gaussian(add)
                    else:
                        break
                if coefficient > 0:
                    for add in range(1, 30):
                        if i + add < len(seq) and j - add >= 0:
                            score = self.paired(seq[i + add], seq[j - add])
                            if score == 0:
                                break
                            else:
                                coefficient = coefficient + score * self.Gaussian(add)
                        else:
                            break
                mat[[i], [j]] = coefficient
        return mat.astype(np.float32)

    @staticmethod
    def Gaussian(x):
        return math.exp(-0.5 * (x * x))

    @staticmethod
    def paired(x, y):
        if x == "A" and y == "U":
            return 2
        elif x == "G" and y == "C":
            return 3
        elif x == "G" and y == "U":
            return 0.8
        elif x == "U" and y == "A":
            return 2
        elif x == "C" and y == "G":
            return 3
        elif x == "U" and y == "G":
            return 0.8
        else:
            return 0


class UfoldDataset(SecondaryStructure):
    paired_feature_extractor = PairFeature()

    @staticmethod
    def _pad_and_mask(x, y, l):
        # for ufold, we need multiples of 2^4, because their NN implementation
        # (4 layers of /2 max-pool and 2x upsampling)
        # (need to maintain even length after all pooling layers)
        # (concat won't work for odd-length, since length will be even after max-pool)
        BLOCK_SIZE = 2**4  # need to be int multiple of this number

        mask = np.ones((l, l))
        if l % BLOCK_SIZE == 0:
            return x, y, mask

        pad_size = BLOCK_SIZE - l % BLOCK_SIZE
        x = np.pad(
            x, ((0, 0), (0, pad_size), (0, pad_size))
        )  # (17, L+pad_size, L+pad_size)
        y = np.pad(y, ((0, pad_size), (0, pad_size)))  # (L+pad_size, L+pad_size)
        mask = np.pad(mask, ((0, pad_size), (0, pad_size)))  # (L+pad_size, L+pad_size)
        return x, y, mask

    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        sequence, one_idx = row[self.seq_col], row[self.pair_col]

        # FIXME replacing non-canonical bases with N
        # since data.py does not support them yet! (some character go out of bound, e.g. 103)
        sequence = sequence.upper()
        sequence = re.sub(r"([^ACGTU])", "N", sequence)

        x_one_hot = fast_one_hot(sequence)
        y = idx2matrix(one_idx, len(sequence))  # (L, 4), (L, L)

        # first 16 features (see paper)
        x_1 = np.kron(x_one_hot, x_one_hot).reshape(
            [len(sequence), len(sequence), 16]
        )  # (L, L, 16)
        x_1 = np.transpose(x_1, [2, 0, 1])  # (16, L, L)

        # the 17th feature
        x_2 = self.paired_feature_extractor.compute_feature(sequence)  # (L, L)

        x = np.concatenate([x_1, x_2[np.newaxis, :]], axis=0)  # (17, L, L)

        x, y, mask = self._pad_and_mask(x, y, len(sequence))

        return {
            "seq": sequence,
            "x": torch.Tensor(x),
            "y": torch.Tensor(y),
            "m": torch.Tensor(mask),
        }


# ==================================================================================================================
# Lightning data module
# ==================================================================================================================


class BaseDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers=0,
        prefetch_factor: int = 4,
        support_ufold: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.support_ufold = support_ufold

        if self.support_ufold:
            self.dataset_class = UfoldDataset
            self.collate_fn = collate_with_mask
        else:
            self.dataset_class = SecondaryStructure
            self.collate_fn = collate_rna_data


class BPRNADatasetModule(BaseDatasetModule):
    def __init__(
        self,
        batch_size: int,
        num_workers=0,
        prefetch_factor: int = 4,
        support_ufold: bool = False,
    ):
        super().__init__(
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            support_ufold=support_ufold,
        )

        df_bprna = load_bprna()
        df_train = df_bprna[df_bprna["split"] == "TR0"]
        df_val = df_bprna[df_bprna["split"] == "VL0"]
        df_test = df_bprna[df_bprna["split"] == "TS0"]

        self.dataset_train = self.dataset_class(df_train)
        self.dataset_val = self.dataset_class(df_val)
        self.dataset_test = self.dataset_class(df_test)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )


class RNASDBDatasetModule(BaseDatasetModule):
    def __init__(
        self,
        batch_size: int,
        split_name: str,
        max_seq_len: int = 450,  # this allows ResNet to work with batch size 2
        num_workers=0,
        prefetch_factor: int = 4,
        seed_only: bool = False,
        weighted_sampling: bool = False,
        test_batch_size: int = 1,
        support_ufold: bool = False,
    ):
        super().__init__(
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            support_ufold=support_ufold,
        )

        if seed_only and weighted_sampling:
            raise ValueError("Cannot use both seed_only and weighted_sampling")

        self.split_name = split_name
        self.max_seq_len = max_seq_len
        self.seed_only = seed_only
        self.weighted_sampling = weighted_sampling
        self.test_batch_size = test_batch_size

    def setup(self, stage: str):
        # Common setup for RNASDB
        split_train_val = True if stage == "fit" else False
        self.rnasdb = RNASDB.initialize_presplit(
            self.split_name, seed_only=self.seed_only, split_train_val=split_train_val
        )
        if self.max_seq_len is not None:
            self.rnasdb.filter_seq_len(self.max_seq_len)

        if stage == "fit":
            self.dataset_train = self.dataset_class(self.rnasdb.df_train)
            self.dataset_val = self.dataset_class(self.rnasdb.df_val)
        elif stage == "test":
            self.dataset_test = self.dataset_class(self.rnasdb.df_test)

            # Setup orthogonal test sets
            self.df_archiveII = load_archiveII(split_name=f"test_{self.split_name}")
            self.dataset_archiveII = self.dataset_class(self.df_archiveII)

    def train_dataloader(self):
        if self.weighted_sampling:
            print("Using weighted sampling")
            # TODO: test this
            weights = self._calc_sampling_weights(self.rnasdb.df_train)
            sampler = WeightedRandomSampler(
                weights, len(self.rnasdb.df_train), replacement=True
            )
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )
        else:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )

    def val_dataloader(self):
        if self.weighted_sampling:
            weights = self._calc_sampling_weights(self.rnasdb.df_val)
            sampler = WeightedRandomSampler(
                weights, len(self.rnasdb.df_val), replacement=True
            )
            return DataLoader(
                self.dataset_val,
                batch_size=self.test_batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )
        else:
            return DataLoader(
                self.dataset_val,
                batch_size=self.test_batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )

    def test_dataloader(self):
        dataloader_rnasdb = DataLoader(
            self.dataset_test,
            batch_size=self.test_batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,  # TODO: fix this after
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
        dataloader_archiveII = DataLoader(
            self.dataset_archiveII,
            batch_size=1,  # for the 'cleanest' evaluation, best to not batch
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

        return [dataloader_rnasdb, dataloader_archiveII]

    def predict_dataloader(self):
        return self.test_dataloader()  # TODO same for now!!

    def _calc_sampling_weights(self, df):
        num_clusters = df.groupby("rfam_family")["cluster_id"].nunique()
        num_clusters = num_clusters.rename("num_clusters")

        df_merged = df.merge(num_clusters, on="rfam_family", how="left")
        assert len(df_merged) == len(df)

        weights = (1 / (df_merged["num_clusters"] * df_merged["cluster_size"])).values

        return torch.from_numpy(weights)


class ArchiveIIDatasetModule(BaseDatasetModule):
    def __init__(
        self,
        batch_size: int,
        split_name: str,
        max_seq_len: Optional[int] = None,
        num_workers=0,
        prefetch_factor: int = 4,
        support_ufold: bool = False,
    ):
        super().__init__(
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            support_ufold=support_ufold,
        )

        self.split_name = split_name
        self.max_seq_len = max_seq_len

        df = load_archiveII(max_seq_len=max_seq_len)
        df_train = df[~(df["split"] == f"test_{self.split_name}")]
        df_test = df[df["split"] == f"test_{self.split_name}"]
        assert len(df) == len(df_train) + len(df_test)

        df_train, df_val = train_test_split(df_train, test_size=0.2)

        self.dataset_train = self.dataset_class(df_train)
        self.dataset_val = self.dataset_class(df_val)
        self.dataset_test = self.dataset_class(df_test)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
