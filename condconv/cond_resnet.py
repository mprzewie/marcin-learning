from typing import Optional, Callable, Type, List, Any

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock, model_urls

from cond_layers import CondConv, CondBatchNorm, CondSequential
from condconv.resnet import ConfigurableResNet


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return CondConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> CondConv:
    """3x3 convolution with padding"""
    return CondConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicResnetCondBlock(BasicBlock):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        nn.Module.__init__(self)
        if norm_layer is None:
            norm_layer = CondBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, k: int) -> Tensor:
        identity = x

        out = self.conv1(x, k=k)
        out = self.bn1(out, k=k)
        out = self.relu(out)

        out = self.conv2(out, k=k)
        out = self.bn2(out, k=k)

        if self.downsample is not None:
            identity = self.downsample(x, k=k)

        out += identity
        out = self.relu(out)

        return out



class CondResnet(ConfigurableResNet):
    def __init__(
        self,
        block: Type[BasicResnetCondBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_planes: int = 64
    ):
        if norm_layer is None:
            norm_layer = CondBatchNorm

        super().__init__(
            block=block, layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            in_planes=in_planes
        )

    def _make_layer(self, block: Type[BasicResnetCondBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:


        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = CondSequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return CondSequential(*layers)


    def forward(self, x: Tensor, k: int, return_intermediate: bool = False) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x, k=k)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = x = self.layer1(x, k=k)
        l2 = x = self.layer2(x, k=k*2)
        l3 = x = self.layer3(x, k=k*4)
        l4 = x = self.layer4(x, k=k*8)

        x = self.avgpool(x)
        penultimate = x = torch.flatten(x, 1)
        x = self.fc(x)

        intermediate = {
            "layer1": l1,
            "layer2": l2,
            "layer3": l3,
            "layer4": l4,
            "penultimate": penultimate
        }
        if return_intermediate:
            return x, intermediate

        return x

def _cond_resnet(
    arch: str,
    block: Type[BasicResnetCondBlock],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> CondResnet:
    model = CondResnet(block, layers, **kwargs)
    if pretrained:
        raise NotImplementedError()
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def cond_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cond_resnet("resnet18", BasicResnetCondBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


