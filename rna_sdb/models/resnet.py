from typing import Optional

import einops
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import BoolTensor, Tensor
from torchmetrics.classification import BinaryF1Score

from rna_sdb.models.base import BaseModule, MaskedConv2d

# ==================================================================================================================
# Migrated from https://github.com/ml4bio/RNA-FM/blob/edffd7a620153e201959c3b1682760086817cd9e/fm/downstream/pairwise_predictor/pairwise_concat.py#L163-L232  # noqa
# with minor cleanup
# ==================================================================================================================


class ResBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        dilation: int = 1,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(
            inplanes, planes, kernel_size=3, bias=False, padding="same"
        )

        self.dropout = nn.Dropout(p=0.3)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d(
            planes, planes, kernel_size=3, dilation=dilation, bias=False, padding="same"
        )

    def forward(self, x: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out, padding_mask=padding_mask)

        out = self.dropout(out)

        out = self.relu2(out)
        out = self.conv2(out, padding_mask=padding_mask)

        out += identity

        return out


class ResNet(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_classes: int = 1):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        res_layers = []
        for i in range(self.num_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(ResBlock(self.dim, self.dim, dilation=dilation))
        self.res_layers = nn.ModuleList(res_layers)

        self.output_layer = MaskedConv2d(
            dim, num_classes, kernel_size=1, masking_after=True, bias=False
        )

    def forward(self, x: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        for layer in self.res_layers:
            x = layer(x, padding_mask=padding_mask)

        x = self.output_layer(x)

        return x


class PairwiseConcatWithResNet(nn.Module):
    """
    contact predictor with pairwise concat + resnet
    reproduce from msa tranformer
    """

    def __init__(
        self,
        dim: int = 64,  # from the original code
        dim_in: int = 4,
        num_layers: int = 32,  # from the original code
        num_classes: int = 1,
        symmetric=True,
    ):
        super().__init__()

        self.dim = dim
        self.dim_in = dim_in
        self.symmetric = symmetric

        self.embed_dim_out = dim // 2
        if dim % 2 != 0:
            raise ValueError("dim must be divisible by 2")
        self.embed_proj = nn.Linear(dim_in, self.embed_dim_out)

        self.resnet = ResNet(dim=dim, num_layers=num_layers, num_classes=num_classes)

    def compute_padding_mask(self, seq: Tensor, dim: int = 1) -> Tensor:
        """Compute padding mask for the input sequence.
        Assume seq is zero-padded, and mask is 'True' for these padded positions.

        Args:
            seq (torch.Tensor): Input sequence tensor
            dim (int): base one-hot-encoding dimension. Defaults to 1.

        Returns:
            torch.Tensor: Padding mask with 'True' for padded positions
        """
        mask = torch.sum(seq, dim=dim) == 0

        return mask

    def outer_concat(self, x: Tensor) -> Tensor:
        """Compute 2D outer concatenation of the input 1D feature map.

        Each position i,j of the 2D output feature map is the concatenation of
        the i and j of 1D input feature map.

        Args:
            x (torch.Tensor): Input 1D feature map with shape (batch, L, d)
            mask (torch.Tensor): Padding mask for the input sequence

        Returns:
            torch.Tensor: 2D outer concatenated feature map
                            with shape (batch, 2 * d, L, L)
        """
        x = einops.rearrange(x, "b l d -> b d l")
        L = x.shape[-1]

        x1 = einops.repeat(x, "b d l -> b d l_new l", l_new=L)
        x2 = einops.repeat(x, "b d l -> b d l l_new", l_new=L)

        x_2d = torch.cat([x1, x2], 1)  # (batch, 2 * d, L, L)

        return x_2d

    def compute_padding_mask_2d(self, mask: BoolTensor) -> BoolTensor:
        """Compute 2D padding mask from 1D mask.

        Args:
            mask (torch.Tensor): 1D padding mask with shape (batch, L)
                                    'True' for padded positions

        Returns:
            torch.Tensor: 2D padding mask with shape (batch, L, L)
                            'True' for padded positions
        """
        assert len(mask.shape) == 2  # (batch, L)

        mask = mask.unsqueeze(2)  # (batch, L, 1)
        mask_2d = self.outer_concat(mask)  # (batch, 2, L, L)
        mask_2d = torch.sum(mask_2d, dim=1) != 0  # (batch, L, L)

        return mask_2d

    def forward(self, embeddings: Tensor) -> Tensor:
        if len(embeddings.shape) != 3 or embeddings.shape[2] != self.dim_in:
            raise ValueError(
                f"Input sequence shape should be (batch, L, {self.dim_in})"
            )

        x = self.embed_proj(embeddings)

        mask = self.compute_padding_mask(x, dim=2)
        x_2d = self.outer_concat(x)
        mask_2d = self.compute_padding_mask_2d(mask)

        output = self.resnet(x_2d, padding_mask=mask_2d)

        if self.symmetric:
            upper_triangular_output = torch.triu(output)
            lower_triangular_output = torch.triu(output, diagonal=1).permute(0, 1, 3, 2)
            output = upper_triangular_output + lower_triangular_output
            output = torch.squeeze(output, dim=1)

        return output, mask_2d


# ==================================================================================================================
# PyTorch Lightning module
# ==================================================================================================================


class ResNetModule(BaseModule):
    def __init__(
        self,
        num_layers: int = 32,  # from the original code
        threshold: float = 0.5,
    ):
        super().__init__(threshold=threshold)

        self.model = PairwiseConcatWithResNet(num_layers=num_layers)
