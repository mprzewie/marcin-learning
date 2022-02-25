from copy import deepcopy
from unittest import TestCase

import torch
from torch.nn import CrossEntropyLoss
from torch.testing import assert_equal, assert_close

from cond_layers import CondConv, CondLinear, CondBatchNorm
from cond_resnet import cond_resnet18, FC_FOR_CHANNELS, MAIN_FC_KS


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
        _, intermediate_full = r(img, return_intermediate=True)

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

    def test_fc_for_channels(self):
        img = torch.randn(size=(1, 3, 10, 10))
        fc_for_channels = [1, 8, 10, 24, 32]

        r = cond_resnet18(fc_for_channels=fc_for_channels)
        r.eval()

        n_channels = 64

        _, intermediate_full = r(img, return_intermediate=True)

        for n in range(1, n_channels + 1):
            _, intermediate_zero = r(img, full_k=n, return_intermediate=True)
            for k, v in intermediate_zero[FC_FOR_CHANNELS].items():
                self.assertLessEqual(k, n)
                assert_equal(
                    intermediate_full[FC_FOR_CHANNELS][k], v
                )

    def test_main_fc_ks(self):
        img = torch.randn(size=(1, 3, 10, 10))
        main_fc_ks = [1, 8, 10, 24, 32]

        r = cond_resnet18()
        r.eval()
        _, intermediate_full = r(img, return_intermediate=True, main_fc_ks=main_fc_ks)

        for n in main_fc_ks:
            out = r(img, full_k=n)
            assert_equal(out, intermediate_full[MAIN_FC_KS][n])


class TestLukaszMarcin(TestCase):
    def test_lukasz_marcin_approach_forward_pass(self):
        """Test if passing image through resnet with K channels is the same as
        zeroing out K*8 channels before the final FC layer"""

        img = torch.randn(size=(1, 3, 10, 10))
        r = cond_resnet18()
        r.eval()

        n_channels = 64
        _, intermediate_full = r(img, return_intermediate=True)

        emb_full = intermediate_full["embedding"]
        for k in range(1, n_channels + 1):
            k_cls, k_intermediate = r(img, k=k, return_intermediate=True)
            emb_k = k_intermediate["embedding"]

            mask = torch.ones_like(emb_full)
            k_out = k * 8
            mask[0, k_out:] = 0
            emb_mask = emb_full * mask

            assert_equal(emb_mask, emb_k)
            assert_equal(k_cls, r.fc(emb_k))
            assert_equal(k_cls, r.fc(emb_mask))

    def test_lukasz_marcin_approach_backward_pass(self):
        img = torch.randn(size=(2, 3, 10, 10))
        r1 = cond_resnet18(norm_layer=None)
        r2 = cond_resnet18(norm_layer=None)
        r2.load_state_dict(r1.state_dict())

        assert_equal(r1.fc.weight, r2.fc.weight)

        r1.eval()
        r2.eval()

        ks = list(range(1, 64, 6))
        loss_fn = CrossEntropyLoss()
        y_true = torch.tensor([1, 2])

        # marcin approach
        loss_1 = 0

        for k in ks:
            y_pred = r1(img, full_k=k)
            l = loss_fn(y_pred, y_true)
            loss_1 += l

        # lukasz approach
        loss_2 = 0
        _, intermediate_full = r2(img, return_intermediate=True)
        emb_full = intermediate_full["embedding"]

        for k in ks:
            mask = torch.ones_like(emb_full)
            k_out = k * 8
            mask[torch.arange(len(mask)), k_out:] = 0
            emb_mask = emb_full * mask
            y_pred = r2.fc(emb_mask)
            l = loss_fn(y_pred, y_true)
            loss_2 += l

        assert_equal(loss_1, loss_2)
        loss_1.backward()
        loss_2.backward()

        for ((n1, p1), (n2, p2)) in zip(r1.named_parameters(), r2.named_parameters()):
            self.assertEqual(n1, n2)
            assert_equal(p1, p2, msg=n1)

            if p1.grad is None:
                self.assertIsNone(p2.grad)
            else:
                assert_close(p1.grad, p2.grad, msg=n1)
