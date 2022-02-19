from unittest import TestCase

import torch
from torch.testing import assert_equal, assert_close

from cond_layers import CondConv, CondLinear, CondBatchNorm
from cond_resnet import cond_resnet18


class TestCondConv(TestCase):

    def test_condconv_same_in_out(self):
        n_channels = 5
        cc = CondConv(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1)
        img = torch.randn(size=(1, n_channels, 10, 10))
        out_full = cc(img, k=n_channels)
        print("The output of N-convolution on full image should be equal to N-convolution on N-image")

        for n in range(1, n_channels + 1):
            mask = torch.ones_like(img)
            mask[0, n:] = 0

            img_zero = img * mask

            out = cc(img, k=n)
            out_zero = cc(img_zero, k=n)

            assert_equal(out, out_zero)
            assert_equal(out_zero[0, n:], torch.zeros_like(out_zero[0, n:]))
            assert_equal(out_zero[0, :n], out_full[0, :n])

    def _test_cond_conv_variable_channels(self, in_channels: int, out_channels: int):

        cc = CondConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        img = torch.randn(size=(1, in_channels, 10, 10))
        out_full = cc(img, k=in_channels)
        print("The output of N-convolution on full image should be equal to N-convolution on N-image")

        for n in range(1, in_channels + 1):
            n_out = n * cc.channel_multiplication

            mask = torch.ones_like(img)
            mask[0, n:] = 0

            img_zero = img * mask

            out = cc(img, k=n)
            out_zero = cc(img_zero, k=n)

            assert_equal(out, out_zero)
            assert_equal(out_zero[0, n_out:], torch.zeros_like(out_zero[0, n_out:]))
            assert_equal(out_zero[0, :n_out], out_full[0, :n_out])

    def test_condconv_out_larger_than_in(self):
        for in_channels in [2, 4, 7]:
            for mult in [1, 2, 5, 10]:
                self._test_cond_conv_variable_channels(in_channels, in_channels * mult)

class TestCondLinear(TestCase):
    def test_cond_linear_same_in_out(self):
        in_size = 5

        cl = CondLinear(in_features=in_size, out_features=in_size)

        v = torch.randn(size=(1, in_size))
        out_full = cl(v, k=in_size)

        for n in range(1, in_size + 1):
            mask = torch.ones_like(v)
            mask[n:] = 0
            v_zero = v * mask

            out = cl(v, k=n)
            out_zero = cl(v_zero, k=n)

            assert_equal(
                out, out_zero
            )
            assert_equal(
                out_zero[0, n:], torch.zeros_like(out_zero[0, n:])
            )
            assert_equal(
                out_zero[0, :n], out_full[0, :n]
            )

    def _test_cond_linear_var_channels(self, in_size: int, out_size: int):
        cl = CondLinear(in_features=in_size, out_features=out_size)

        v = torch.randn(size=(1, in_size))
        out_full = cl(v, k=in_size)

        for n in range(1, in_size + 1):
            n_out = n * cl.feature_multiplication
            mask = torch.ones_like(v)
            mask[n:] = 0
            v_zero = v * mask

            out = cl(v, k=n)
            out_zero = cl(v_zero, k=n)

            assert_equal(
                out, out_zero
            )
            assert_equal(
                out_zero[0, n_out:], torch.zeros_like(out_zero[0, n_out:])
            )
            assert_equal(
                out_zero[0, :n_out], out_full[0, :n_out]
            )

    def test_cond_linear_in_larger_than_out(self):
        for in_features in [2, 4, 7]:
            for mult in [1, 2, 5, 10]:
                self._test_cond_linear_var_channels(in_features, in_features * mult)

class TestCondBatchNorm(TestCase):

    def test_cond_batchnorm(self):
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

            assert_equal(
                out, out_zero
            )

            assert_equal(
                out_zero[0, n:], torch.zeros_like(out_zero[0, n:])
            )

            assert_equal(
                out_zero[0, :n], out_full[0, :n]
            )

    def test_tril_repeated(self):
        x = torch.tril(torch.ones(3, 3))
        assert_equal(
            x, torch.tensor([
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]
            ]).float()
        )

        x_rep_0 = x.repeat_interleave(2, 0)

        assert_equal(
            x_rep_0,
            torch.tensor([
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
            ]).float()
        )

        x_rep_1 = x.repeat_interleave(2, 1)

        assert_equal(
            x_rep_1,
            torch.tensor([
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1]
            ]).float()
        )


class TestCondResNet(TestCase):
    def test_resnet18(self):
        img = torch.randn(size=(1, 3, 10, 10))
        r = cond_resnet18()
        r.eval()

        n_channels = 64
        _, intermediate_full = r(img, k=n_channels, return_intermediate=True)

        for n in range(1, n_channels + 1):
            _, intermediate_zero = r(img, k=n, return_intermediate=True)
            for (k, v) in intermediate_zero.items():

                intermediate_n_channels = v.shape[1]
                self.assertEqual(
                    intermediate_n_channels // n_channels,
                    intermediate_n_channels / n_channels
                )
                chan_mult = intermediate_n_channels // n_channels
                n_out = n * chan_mult

                assert_equal(v[0, n_out:], torch.zeros_like(v[0, n_out:]))
                assert_equal(v[0, :n_out], intermediate_full[k][0, :n_out])
