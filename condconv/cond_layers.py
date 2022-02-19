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

        assert (out_channels / in_channels) == (out_channels // in_channels), f"{in_channels=} must be a divisor of {out_channels=}"
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
        )
        self.channel_multiplication = out_channels // in_channels
        c_out, c_in, kh, kw = self.weight.shape
        self.w_mask = nn.Parameter(
            torch.tril(
                torch.ones(c_in, c_in)
            ).repeat_interleave(
                self.channel_multiplication,
                dim=0
            ).reshape(c_out, c_in, 1, 1),
            requires_grad=False
        )
        self.b_mask = nn.Parameter(
            torch.ones_like(self.bias),
            requires_grad=False
        ) if bias else None

    def forward(self, input: Tensor, *, k: int) -> Tensor:
        k_out = k * self.channel_multiplication
        wm = torch.ones_like(self.w_mask)
        wm[k_out:] = 0

        if self.bias is not None:
            bm = torch.ones_like(self.b_mask)
            bm[k_out:] = 0

        return self._conv_forward(
            input,
            self.weight * self.w_mask * wm,
            self.bias * self.b_mask * bm if self.bias is not None else self.bias
        )


class CondLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        assert (out_features / in_features) == (out_features // in_features), f"{in_features=} must be a divisor of {out_features=}"
        super().__init__(in_features, out_features)

        self.feature_multiplication = out_features // in_features

        self.w_mask = nn.Parameter(
            torch.tril(
                torch.ones(in_features, in_features)
            ).repeat_interleave(
                self.feature_multiplication,
                dim=0
            ), requires_grad=False)
        self.b_mask = nn.Parameter(torch.ones_like(self.bias), requires_grad=False)

    def forward(self, input: Tensor, *, k: int) -> Tensor:
        k_out = k * self.feature_multiplication
        wm = torch.ones_like(self.w_mask)
        bm = torch.ones_like(self.b_mask)
        wm[k_out:] = 0
        bm[k_out:] = 0

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

