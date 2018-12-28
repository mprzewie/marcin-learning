import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        # downsampling
        self.dc1 = conv_block(3, 64)
        self.mp1 = nn.MaxPool2d(2)

        self.dc2 = conv_block(64, 128)
        self.mp2 = nn.MaxPool2d(2)

        self.dc3 = conv_block(128, 256)
        self.mp3 = nn.MaxPool2d(2)

        self.dc4 = conv_block(256, 512)
        self.mp4 = nn.MaxPool2d(2)

        # bottom
        self.bc = conv_block(512, 1024)
        # upsampling

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.uc1 = conv_block(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.uc2 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.uc3 = conv_block(256, 256)

        self.up4 = nn.ConvTranspose2d(256, 192, 2, 2)
        self.uc4 = conv_block(256, 256)

        self.oc = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

    def forward(self, images: torch.Tensor):
        #down
        dc1o = self.dc1(images)
        mp1o = self.mp1(dc1o)

        dc2o = self.dc2(mp1o)
        mp2o = self.mp2(dc2o)

        dc3o = self.dc3(mp2o)
        mp3o = self.mp3(dc3o)

        dc4o = self.dc4(mp3o)
        mp4o = self.mp4(dc4o)

        #bottom
        bco = self.bc(mp4o)

        #up
        up1o = self.up1(bco)
        uc1in = torch.cat([up1o, dc4o], 1)
        uc1o = self.uc1(uc1in)

        up2o = self.up2(uc1o)
        uc2in = torch.cat([up2o, dc3o], 1)
        uc2o = self.uc2(uc2in)

        up3o = self.up3(uc2o)
        uc3in = torch.cat([up3o, dc2o], 1)
        uc3o = self.uc3(uc3in)

        up4o = self.up4(uc3o)
        uc4in = torch.cat([up4o, dc1o], 1)
        uc4o = self.uc4(uc4in)
        output = self.oc(uc4o)

        return output


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
    )
