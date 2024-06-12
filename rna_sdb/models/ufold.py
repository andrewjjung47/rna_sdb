"""UFold re-implemented from
https://github.com/uci-cbcl/UFold/blob/3c92fa184ae66e385214f3e4c1da6cf9bfd667f5/Network.py#L40
"""

import math
import random
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import BoolTensor, Tensor
from torch.utils.data import DataLoader, Dataset

from rna_sdb.models.base import BaseModule, MaskedConv2d

# ==================================================================================================================
# UFold model (new)
# ==================================================================================================================


class UNetConvBlock(nn.Module):
    def __init__(self, in_c, out_c, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.conv_1 = MaskedConv2d(
            in_c, out_c, kernel_size=3, bias=False, padding="same"
        )
        self.norm_1 = norm_layer(out_c)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2 = MaskedConv2d(
            out_c, out_c, kernel_size=3, bias=False, padding="same"
        )
        self.norm_2 = norm_layer(out_c)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        x = self.conv_1(x, padding_mask=padding_mask)
        x = self.norm_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x, padding_mask=padding_mask)
        x = self.norm_2(x)
        x = self.relu_2(x)

        return x


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.conv_block = UNetConvBlock(in_c, out_c, norm_layer=norm_layer)

    def forward(self, x: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        x = self.max_pool(x)

        if padding_mask is not None:
            padding_mask = self.max_pool(
                padding_mask
            )  # need to downsample padding mask too

        x = self.conv_block(x, padding_mask=padding_mask)

        return x, padding_mask


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert in_c % 2 == 0

        self.conv_transpose = nn.ConvTranspose2d(
            in_c, in_c // 2, kernel_size=2, stride=2
        )
        self.conv_block = UNetConvBlock(in_c, out_c, norm_layer=norm_layer)

    def forward(
        self, x: Tensor, x_skip: Tensor, padding_mask: Optional[BoolTensor] = None
    ) -> Tensor:
        if x.shape[-1] * 2 + 1 == x_skip.shape[-1]:
            x = self.conv_transpose(x, x_skip.size())
        else:
            x = self.conv_transpose(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_block(x, padding_mask=padding_mask)

        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, num_input_ch=17, conv_filters=[32, 64, 128, 256, 512]):
        super(U_Net, self).__init__()

        # TODO rewrite this code if we need to expose more HPs like n_blocks
        assert len(conv_filters) == 5  # rest of code hard-coded for 5 blocks

        self.enc_blocks = nn.ModuleList(
            [
                UNetConvBlock(num_input_ch, conv_filters[0]),
                UNetEncoderBlock(conv_filters[0], conv_filters[1]),
                UNetEncoderBlock(conv_filters[1], conv_filters[2]),
                UNetEncoderBlock(conv_filters[2], conv_filters[3]),
                UNetEncoderBlock(conv_filters[3], conv_filters[4]),
            ]
        )

        self.dec_blocks = nn.ModuleList(
            [
                UNetDecoderBlock(conv_filters[4], conv_filters[3]),
                UNetDecoderBlock(conv_filters[3], conv_filters[2]),
                UNetDecoderBlock(conv_filters[2], conv_filters[1]),
                UNetDecoderBlock(conv_filters[1], conv_filters[0]),
            ]
        )

        self.logit = nn.Conv2d(conv_filters[0], 1, kernel_size=1)

    def forward(self, x: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        # encoding path
        x1 = self.enc_blocks[0](x, padding_mask=padding_mask)
        x2, padding_mask_2 = self.enc_blocks[1](x1, padding_mask=padding_mask)
        x3, padding_mask_3 = self.enc_blocks[2](x2, padding_mask=padding_mask_2)
        x4, padding_mask_4 = self.enc_blocks[3](x3, padding_mask=padding_mask_3)
        x5, _ = self.enc_blocks[4](x4, padding_mask=padding_mask_4)

        d5 = self.dec_blocks[0](x5, x4, padding_mask=padding_mask_4)
        d4 = self.dec_blocks[1](d5, x3, padding_mask=padding_mask_3)
        d3 = self.dec_blocks[2](d4, x2, padding_mask=padding_mask_2)
        d2 = self.dec_blocks[3](d3, x1, padding_mask=padding_mask)

        d1 = self.logit(d2).squeeze(1)

        return (torch.transpose(d1, -1, -2) + d1) / 2


class U_Net_old(nn.Module):
    def __init__(self, num_input_ch=17, conv_filters=[32, 64, 128, 256, 512]):
        super(U_Net, self).__init__()

        # TODO rewrite this code if we need to expose more HPs like n_blocks
        assert len(conv_filters) == 5  # rest of code hard-coded for 5 blocks

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=num_input_ch, ch_out=conv_filters[0])
        self.Conv2 = conv_block(ch_in=conv_filters[0], ch_out=conv_filters[1])
        self.Conv3 = conv_block(ch_in=conv_filters[1], ch_out=conv_filters[2])
        self.Conv4 = conv_block(ch_in=conv_filters[2], ch_out=conv_filters[3])
        self.Conv5 = conv_block(ch_in=conv_filters[3], ch_out=conv_filters[4])

        # Alice: I don't know how ufold github code doesn't blow up (for generic conv filter sizes) ...
        # the ch_in of the de-conv path needs to take into account skip connection
        self.Up5 = up_conv(ch_in=conv_filters[4], ch_out=conv_filters[3])
        self.Up_conv5 = conv_block(ch_in=conv_filters[3] * 2, ch_out=conv_filters[3])

        self.Up4 = up_conv(ch_in=conv_filters[3], ch_out=conv_filters[2])
        self.Up_conv4 = conv_block(ch_in=conv_filters[2] * 2, ch_out=conv_filters[2])

        self.Up3 = up_conv(ch_in=conv_filters[2], ch_out=conv_filters[1])
        self.Up_conv3 = conv_block(ch_in=conv_filters[1] * 2, ch_out=conv_filters[1])

        self.Up2 = up_conv(ch_in=conv_filters[1], ch_out=conv_filters[0])
        self.Up_conv2 = conv_block(ch_in=conv_filters[0] * 2, ch_out=conv_filters[0])

        self.Conv_1x1 = nn.Conv2d(
            conv_filters[0], 1, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)

        return torch.transpose(d1, -1, -2) * d1


class UFoldTrainer(BaseModule):
    def __init__(
        self,
        conv_filters: List[int] = [
            32,
            64,
            128,
            256,
            512,
        ],
        threshold: float = 0.5,
    ):
        super().__init__(threshold=threshold)

        self.model = U_Net(conv_filters=conv_filters)
        # loss weighting: https://github.com/uci-cbcl/UFold/blob/3c92fa184ae66e385214f3e4c1da6cf9bfd667f5/ufold_train.py#L34
        # move to device with https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#init-tensors-using-tensor-to-and-register-buffer
        # self.loss_fn = torch.nn.BCEWithLogitsLoss(
        #     pos_weight=torch.Tensor([300]).to(self.device), reduction="none"
        # )  # for masking

    def _step(self, batch):
        if "m" in batch:  # with masking
            x, y, mask = batch["x"], batch["y"], batch["m"]
            y_hat = self.model(x)

            loss = self._masked_loss_fn(y_hat, y, mask)
        else:  # no masking
            x, y = batch["x"], batch["y"]
            y_hat = self.model(x)
            loss_arr = self.loss_fn(y_hat, y)
            loss = loss_arr.mean()

        return loss, y_hat, y

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch["x"], batch["y"], batch["m"]
        seq = batch["seq"]

        # only support batch size = 1 for now
        assert len(seq) == 1
        seq = seq[0]

        y_hat = self.model(x)
        # crop to seq len (recall that we need to pad for NN due to the hard-coded down/up sampling layers in ufold)
        assert y_hat.shape[0] == 1
        assert len(y_hat.shape) == 3
        assert y_hat.shape[1] == y_hat.shape[2]
        y_hat = y_hat[:, : len(seq), : len(seq)]

        # post-processing
        # from https://github.com/uci-cbcl/UFold/blob/3c92fa184ae66e385214f3e4c1da6cf9bfd667f5/ufold_predict.py#L182
        # FIXME hard-coded to postprocess_new, we should also support postprocess_new_nc!
        y_hat_pp = postprocess_new(
            torch.unsqueeze(y_hat, 0),  # fake batch
            torch.unsqueeze(one_hot_ufold(seq), 0),  # fake batch
            0.01,
            0.1,
            100,
            1.6,
            True,
            1.5,
        )
        y_hat_bin = (y_hat_pp > 0.5).float()

        # verify shapes and remove batch dim
        assert y_hat.shape[0] == 1
        # I don't why ufold post-processing code adds extra singleton dim (can review their code and figure out, but I don't have time now)
        assert y_hat_pp.shape[0] == 1
        assert y_hat_pp.shape[1] == 1
        assert y_hat_bin.shape[0] == 1
        assert y_hat_bin.shape[1] == 1
        y_hat = torch.squeeze(y_hat, dim=0)
        y_hat_pp = torch.squeeze(y_hat_pp)
        y_hat_bin = torch.squeeze(y_hat_bin)

        # return before and after post-processing
        return y_hat, y_hat_pp, y_hat_bin


# ==================================================================================================================
# UFold data loader
# ==================================================================================================================


## post-processing ##

# for ufold one-hot, order: AUCG
# needed for re-using their post-processing function
BASE_TO_ONE_HOT_LABEL_UFOLD = np.zeros([90, 4], dtype=np.float32)
BASE_TO_ONE_HOT_LABEL_UFOLD[[ord(base) for base in "ATCG"]] = np.eye(
    4
)  # note difference in order!
BASE_TO_ONE_HOT_LABEL_UFOLD[ord("U")] = BASE_TO_ONE_HOT_LABEL_UFOLD[ord("T")]


def one_hot_ufold(seq):
    bseq = seq.encode("ascii")
    encoding = BASE_TO_ONE_HOT_LABEL_UFOLD[memoryview(bseq)]
    return torch.tensor(encoding, dtype=torch.float)


# from https://github.com/uci-cbcl/UFold/blob/3c92fa184ae66e385214f3e4c1da6cf9bfd667f5/ufold/postprocess.py


def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return au_ua + cg_gc + ug_gu


def constraint_matrix_batch_addnc(x):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    ## add non-canonical pairs
    ac = torch.matmul(base_a.view(batch, length, 1), base_c.view(batch, 1, length))
    ac_ca = ac + torch.transpose(ac, -1, -2)
    ag = torch.matmul(base_a.view(batch, length, 1), base_g.view(batch, 1, length))
    ag_ga = ag + torch.transpose(ag, -1, -2)
    uc = torch.matmul(base_u.view(batch, length, 1), base_c.view(batch, 1, length))
    uc_cu = uc + torch.transpose(uc, -1, -2)
    aa = torch.matmul(base_a.view(batch, length, 1), base_a.view(batch, 1, length))
    uu = torch.matmul(base_u.view(batch, length, 1), base_u.view(batch, 1, length))
    cc = torch.matmul(base_c.view(batch, length, 1), base_c.view(batch, 1, length))
    gg = torch.matmul(base_g.view(batch, length, 1), base_g.view(batch, 1, length))
    return au_ua + cg_gc + ug_gu + ac_ca + ag_ga + uc_cu + aa + uu + cc + gg


def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a


def sign(x):
    return (x > 0).type(x.dtype)


def soft_sign(x):
    k = 1
    return 1.0 / (1.0 + torch.exp(-2 * k * x))


def postprocess_new(
    u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)
):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch(x).float()
    # u with threshold
    # equivalent to sigmoid(u) > 0.9
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(u - s) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (
            lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        ).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

        # print
        # if t % 20 == 19:
        #     n1 = torch.norm(lmbd_grad)
        #     grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        #     grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        #     n2 = torch.norm(grad)
        #     print([t, 'norms', n1, n2, aug_lagrangian(u, m, a_hat, lmbd), torch.sum(contact_a(a_hat, u))])

    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a


def postprocess_new_nc(
    u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)
):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch_addnc(x).float()
    # m = 1.0
    # u with threshold
    # equivalent to sigmoid(u) > 0.9
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(u - s) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (
            lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        ).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

        # print
        # if t % 20 == 19:
        #     n1 = torch.norm(lmbd_grad)
        #     grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        #     grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        #     n2 = torch.norm(grad)
        #     print([t, 'norms', n1, n2, aug_lagrangian(u, m, a_hat, lmbd), torch.sum(contact_a(a_hat, u))])

    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a
