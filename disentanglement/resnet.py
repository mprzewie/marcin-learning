'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, detach_residual: bool = False):
        super(BasicBlock, self).__init__()
        
        self.detach_residual = detach_residual
            
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(
            (x.detach() if self.detach_residual else x)
        )
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, detach_residual: bool = False):
        super(Bottleneck, self).__init__()
        
        self.detach_residual = detach_residual

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(
            (x.detach() if self.detach_residual else x)
        )       
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, detach_residual: bool = False,
        dropout: float = 0, cls_len: int = 1
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.detach_residual = detach_residual
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        if cls_len > 1:
            linear_seq = []
            for i in range(cls_len - 1):
                linear_seq.extend([nn.Linear(512*block.expansion, 512*block.expansion), nn.ReLU()])
            linear_seq.append(nn.Linear(512*block.expansion, num_classes))
            self.linear = nn.Sequential(*linear_seq)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)
        self.d4 = nn.Dropout(dropout)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, detach_residual=self.detach_residual))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_activations=False):
        c1 = out = F.relu(self.bn1(self.conv1(x)))
        l1 = out = self.layer1(out)
        l2 = out = self.layer2(self.d1(out))
        l3 = out = self.layer3(self.d2(out))
        l4 = out = self.layer4(self.d3(out))
        out = F.avg_pool2d(out, 4)
        pool = out = out.view(out.size(0), -1)
        out = self.linear(self.d4(out))
        
        if return_activations:
            return dict(
                c1=c1,
                l1=l1,
                l2=l2,
                l3=l3,
                l4=l4, 
                pool=pool,
                out=out
            )
        return out


def ResNet18(num_classes: int = 10, detach_residual: bool = False, dropout: float = 0, cls_len: int = 1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,detach_residual=detach_residual, dropout=dropout, cls_len=cls_len)


def ResNet34(num_classes: int = 10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())