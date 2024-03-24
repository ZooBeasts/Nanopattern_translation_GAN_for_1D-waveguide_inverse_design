from torch import nn
import torch.nn.functional as F
import torch


nc = 1
image_size = 64
ngpu = 1
features_d = 64
features_g = 64
Z_dim = 350
channels_noise = Z_dim




def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class Unetdown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(Unetdown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UnetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True),
                  ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


##########
class GeneratorUnet(nn.Module):
    def __int__(self, in_channel=3, out_channel=5):
        super(GeneratorUnet, self).__init__()

        self.down1 = Unetdown(in_channel, 64, normalize=False)
        self.down2 = Unetdown(64, 128)
        self.down3 = Unetdown(128, 256)
        self.down4 = Unetdown(256, 512, dropout=0.5)
        self.down5 = Unetdown(512, 512, dropout=0.5)
        self.down6 = Unetdown(512, 512, dropout=0.5)
        self.down7 = Unetdown(512, 512, dropout=0.5)
        self.down8 = Unetdown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UnetUp(512, 512, dropout=0.5)
        self.up2 = UnetUp(1024, 512, dropout=0.5)
        self.up3 = UnetUp(1024, 512, dropout=0.5)
        self.up4 = UnetUp(1024, 512, dropout=0.5)
        self.up5 = UnetUp(1024, 256)
        self.up6 = UnetUp(512, 128)
        self.up7 = UnetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channel, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d5)
        u6 = self.up6(u5, d3)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()

        def Discriminator_Block(in_filters, out_filters, normlization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normlization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *Discriminator_Block(in_channel * 2, 64, normlization=False),
            *Discriminator_Block(64, 128),
            *Discriminator_Block(128, 256),
            *Discriminator_Block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class GeneratorConditional(nn.Module):
    def __init__(self):
        super(GeneratorConditional).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, nc, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, points):
        return self.net(points)
