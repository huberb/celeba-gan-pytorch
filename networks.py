from torch import nn
import torchsummary
from torch.optim import Adam


class Discriminator(nn.Module):

    def __init__(self, device='cuda'):
        super(Discriminator, self).__init__()

        channel_dim = 64
        kernel = 4
        stride = 2
        padding = 1

        self.layers = nn.Sequential(
                nn.Conv2d(3, channel_dim,
                          kernel_size=kernel, stride=stride,
                          padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                *self.block(channel_dim, channel_dim * 2,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 2, channel_dim * 4,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 4, channel_dim * 8,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 8, channel_dim * 16,
                            kernel=kernel, stride=stride,
                            padding=padding),
                nn.Conv2d(channel_dim * 16, 1,
                          kernel_size=kernel, stride=1,
                          padding=0),
                nn.Sigmoid()
            ).to(device)

        self.optim = Adam(params=self.layers.parameters(),
                          lr=0.0002, betas=(0.5, 0.999))

    def block(self, input_dim, output_dim, kernel, stride, padding):
        return (
            nn.Conv2d(input_dim, output_dim,
                      kernel_size=kernel, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        return self.layers(x).view(x.shape[0], -1)

    def summary(self):
        torchsummary.summary(self, (3, 128, 128))


class Generator(nn.Module):

    def __init__(self, latent_dim=128, device='cuda'):

        super(Generator, self).__init__()

        channel_dim = 64
        kernel = 4
        stride = 2
        padding = 1

        self.layers = nn.Sequential(
                *self.block(latent_dim, channel_dim * 16,
                            kernel=kernel, stride=stride - 1,
                            padding=padding - 1),
                *self.block(channel_dim * 16, channel_dim * 8,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 8, channel_dim * 4,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 4, channel_dim * 2,
                            kernel=kernel, stride=stride,
                            padding=padding),
                *self.block(channel_dim * 2, channel_dim,
                            kernel=kernel, stride=stride,
                            padding=padding),
                nn.ConvTranspose2d(channel_dim, 3,
                                   kernel_size=kernel, stride=stride,
                                   padding=padding),
                nn.Tanh()
                ).to(device)

        self.optim = Adam(params=self.layers.parameters(),
                          lr=0.0002, betas=(0.5, 0.999))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return self.layers(x)

    def block(self, input_dim, output_dim, kernel, stride, padding):
        return (
            nn.ConvTranspose2d(input_dim, output_dim,
                               kernel_size=kernel, stride=stride,
                               padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
        )

    def summary(self):
        torchsummary.summary(self, (128, 1, 1))
