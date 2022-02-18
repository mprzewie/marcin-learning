from typing import Tuple, Any, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CondConv(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = 1,
                 padding: Union[str, int] = 0,
                 dilation: Tuple[int, ...] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Any = None,
                 dtype: Any = None):
        assert in_channels == out_channels, (in_channels, out_channels)
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
        )
        c_in, c_out, kh, kw = self.weight.shape
        self.w_mask = nn.Parameter(
            torch.tril(torch.ones(c_in, c_out)).reshape(c_in, c_out, 1, 1).to(self.weight.device), requires_grad=False)
        self.b_mask = nn.Parameter(torch.ones_like(self.bias), requires_grad=False) if bias else None

    def forward(self, input: Tensor, *, k: int) -> Tensor:
        wm = torch.ones_like(self.w_mask)
        wm[k:] = 0

        if self.bias is not None:
            bm = torch.ones_like(self.b_mask)
            bm[k:] = 0

        return self._conv_forward(
            input,
            self.weight * self.w_mask * wm,
            self.bias * self.b_mask * bm if self.bias is not None else self.bias
        )


class CondLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        assert in_features == out_features, (in_features, out_features)
        super().__init__(in_features, out_features)
        self.w_mask = nn.Parameter(
            torch.tril(torch.ones_like(self.weight)), requires_grad=False)
        self.b_mask = nn.Parameter(torch.ones_like(self.bias), requires_grad=False)

    def forward(self, input: Tensor, *, k: int) -> Tensor:

        wm = torch.ones_like(self.w_mask)
        bm = torch.ones_like(self.b_mask)
        wm[k:] = 0
        bm[k:] = 0

        return F.linear(input, self.weight * self.w_mask * wm, self.bias * self.b_mask * bm)

class CondBatchNorm(nn.BatchNorm2d):
    def forward(self, input: Tensor, *, k: int) -> Tensor:
        out = super().forward(input)
        b, c, h, w = out.shape
        mask = torch.ones_like(out)
        mask[torch.arange(0, b), k:] = 0
        return out * mask

class CondSequential(nn.Sequential):
    def forward(self, input: torch.Tensor, *, k: int):
        x = input
        for l in self:
            x = l(x, k=k)
        return x

def test_condconv():
    n_channels = 5
    cc = CondConv(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1)
    img = torch.randn(size=(1, n_channels, 10, 10))
    out_full = cc(img, k=n_channels)
    print("The output of N-convolution on full image should be equal to N-convolution on N-image")

    for n in range(1, n_channels + 1):
        mask = torch.zeros_like(img)
        mask[0, :n] = 1

        img_zero = img * mask

        out = cc(img, k=n)
        out_zero = cc(img_zero, k=n)

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
    out_full = cl(v, k=in_size)
    print(out_full.shape == (1, in_size))

    for n in range(1, in_size + 1):
        mask = torch.ones_like(v)
        mask[n:] = 0
        v_zero = v * mask

        out = cl(v, k=n)
        out_zero = cl(v_zero, k=n)

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
    out_full = cb(img, k=n_channels)

    for n in range(1, n_channels + 1):
        mask = torch.zeros_like(img)
        mask[0, :n] = 1

        img_zero = img * mask

        out = cb(img, k=n)
        out_zero = cb(img_zero, k=n)

        print(
            n,
            torch.equal(out, out_zero),
            torch.equal(out_zero[0, n:], torch.zeros_like(out_zero[0, n:])),
            torch.equal(out_zero[0, :n], out_full[0, :n])
        )

if __name__ == "__main__":
    test_condconv()
    test_cond_batchnorm()
    test_cond_linear()
