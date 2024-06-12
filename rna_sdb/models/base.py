from typing import Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import BoolTensor, Tensor
from torchmetrics.classification import BinaryF1Score

# ==================================================================================================================
# Custom conv layers
# ==================================================================================================================


class MaskedConv(nn.Module):
    """Applies zero masking to the padded positions.

    Masking can be applied twice, before and after the convolution, to ensure
    that the padded positions stay as zeros for the convolution input and
    its output stays properly masked as well for the next layer.

    This is because even if the input is properly masked, the output could still
    have non-zero values when conv is run at the padded locations near the boundary
    to the non-masked positions.
    It is possible this can be excessive for many situations.

    Initialize the layer with masking_after=True to apply masking after the convolution.
    (default is masking_after=False)
    """

    def __init__(
        self,
        dim: Literal[1, 2],
        *args,
        masking_after: bool = False,
        mask_val: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.masking_after = masking_after
        self.mask_val = mask_val

        if dim == 1:
            self.conv = nn.Conv1d(*args, **kwargs)
        elif dim == 2:
            self.conv = nn.Conv2d(*args, **kwargs)
        else:
            raise ValueError("Invalid dimension for MaskedConv")

    def forward(self, x: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        """Apply zero masking to the positions indicated by padding_mask.

        Args:
            x: input tensor
            padding_mask: a boolean tensor indicating which positions are padded
                          (elements corresponding to 'True' will be zero masked)

        Returns:
            Tensor: output tensor with zero masking applied
        """
        if x.dim() != 3 and x.dim() != 4:
            raise ValueError(
                "Expected x to have 3 or 4 dimensions (batch, channel, seq_len)"
                " or (batch, channel, seq_len, seq_len)"
            )

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(
                1
            )  # (batch, 1, seq_len) or (batch, 1, seq_len, seq_len)

            x = x.masked_fill(
                padding_mask, self.mask_val
            )  # apply mask_val for the padded positions

        x = self.conv.forward(x)

        if self.masking_after and padding_mask is not None:
            x = x.masked_fill(
                padding_mask, self.mask_val
            )  # apply mask_val for the padded positions

        return x


class MaskedConv1d(MaskedConv):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class MaskedConv2d(MaskedConv):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


# ==================================================================================================================
# Lightning Module
# ==================================================================================================================


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        threshold: float = 0.5,  # TODO: change this hardcoded value?
    ):
        super().__init__()

        self.save_hyperparameters()

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.train_metrics = BinaryF1Score(threshold=threshold)
        self.val_metrics = BinaryF1Score(threshold=threshold)
        self.test_metrics = BinaryF1Score(threshold=threshold)
        self.test_archiveII_metrics = BinaryF1Score(threshold=threshold)

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_label = self._step(batch)

        self.train_metrics.update(
            y_pred, y_label
        )  # TODO: double check if update is correct

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train/f1",
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_label = self._step(batch)
        self.val_metrics.update(y_pred, y_label)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/f1",
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_pred, y_label = self._step(batch)

        if dataloader_idx == 0:
            self.test_metrics.update(y_pred, y_label)

            self.log(
                "test/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "test/f1",
                self.test_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        elif dataloader_idx == 1:
            self.test_archiveII_metrics.update(y_pred, y_label)

            self.log(
                "archiveII/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "archiveII/f1",
                self.test_archiveII_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        else:
            raise ValueError(f"Unknown dataloader index: {dataloader_idx}")

    def _masked_loss_fn(self, y_pred, y_label, mask):
        loss = self.loss_fn(y_pred, y_label)
        loss = torch.sum(loss * mask) / torch.sum(mask)

        return loss

    def _step(self, batch):
        x, y_label = batch

        y_pred, mask_2d = self.model(x)
        mask_2d = ~mask_2d  # flip boolean mask such that 'True' for unpadded positions
        # TODO: decide if it is better to make it the other way? (e.g. consistent with SA)

        loss = self._masked_loss_fn(y_pred, y_label, mask_2d)

        return loss, y_pred, y_label
