from typing import Tuple, Any, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CondConv(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = 1,
                 padding: str = 0,
                 dilation: Tuple[int, ...] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Any = None,
                 dtype: Any = None):
        assert in_channels == out_channels, (in_channels, out_channels)
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=1, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype
        )
        c_in, c_out, kh, kw = self.weight.shape
        self.w_mask = nn.Parameter(
            torch.tril(torch.ones(c_in, c_out)).reshape(c_in, c_out, 1, 1).to(self.weight.device), requires_grad=False)
        self.b_mask = nn.Parameter(torch.ones_like(self.bias), requires_grad=False)

    def forward(self, input: Tensor, k: Optional[int] = None) -> Tensor:
        k = k or self.in_channels

        wm = torch.ones_like(self.w_mask)
        bm = torch.ones_like(self.b_mask)
        wm[k:] = 0
        bm[k:] = 0

        return self._conv_forward(
            input,
            self.weight * self.w_mask * wm,
            self.bias * self.b_mask * bm
        )


class CondLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        assert in_features == out_features, (in_features, out_features)
        super().__init__(in_features, out_features)
        self.w_mask = nn.Parameter(
            torch.tril(torch.ones_like(self.weight)), requires_grad=False)
        self.b_mask = nn.Parameter(torch.ones_like(self.bias), requires_grad=False)

    def forward(self, input: Tensor, k: Optional[int] = None) -> Tensor:
        k = k or self.in_features

        wm = torch.ones_like(self.w_mask)
        bm = torch.ones_like(self.b_mask)
        wm[k:] = 0
        bm[k:] = 0

        return F.linear(input, self.weight * self.w_mask * wm, self.bias * self.b_mask * bm)

class CondBatchNorm(nn.BatchNorm2d):
    def forward(self, input: Tensor, k: Optional[int] = None) -> Tensor:
        k = k or self.num_features
        out = super().forward(input)
        b, c, h, w = out.shape
        mask = torch.ones_like(out)
        mask[torch.arange(0, b), k:] = 0
        return out * mask


def test_condconv():
    n_channels = 5
    cc = CondConv(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1)
    img = torch.randn(size=(1, n_channels, 10, 10))
    out_full = cc(img, n_channels)
    print("The output of N-convolution on full image should be equal to N-convolution on N-image")

    for n in range(1, n_channels + 1):
        mask = torch.zeros_like(img)
        mask[0, :n] = 1

        img_zero = img * mask

        out = cc(img, n)
        out_zero = cc(img_zero, n)

        print(
            n,
            torch.equal(out, out_zero),
            torch.equal(out_zero[0, n:], torch.zeros_like(out_zero[0, n:])),
            torch.equal(out_zero[0, :n], out_full[0, :n])
        )


def test_cond_linear():
    in_size = 5

    cl = CondLinear(in_features=in_size, out_features=in_size)

    v = torch.randn(size=(1, in_size))
    out_full = cl(v)
    print(out_full.shape == (1, in_size))

    for n in range(1, in_size + 1):
        mask = torch.ones_like(v)
        mask[n:] = 0
        v_zero = v * mask

        out = cl(v, n)
        out_zero = cl(v_zero, n)

        print(
            n,
            torch.equal(out, out_zero),
            torch.equal(out_zero[0, n:], torch.zeros_like(out_zero[0, n:])),
            torch.equal(out_zero[0, :n], out_full[0, :n])
        )


def test_cond_batchnorm():
    n_channels = 5
    cb = CondBatchNorm(num_features=n_channels)
    img = torch.randn(size=(1, n_channels, 10, 10))
    out_full = cb(img, n_channels)

    for n in range(1, n_channels + 1):
        mask = torch.zeros_like(img)
        mask[0, :n] = 1

        img_zero = img * mask

        out = cb(img, n)
        out_zero = cb(img_zero, n)

        print(
            n,
            torch.equal(out, out_zero),
            torch.equal(out_zero[0, n:], torch.zeros_like(out_zero[0, n:])),
            torch.equal(out_zero[0, :n], out_full[0, :n])
        )

test_cond_batchnorm()
# test_cond_linear()
