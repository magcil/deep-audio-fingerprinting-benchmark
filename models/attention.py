import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, channels, spect_dims, temp_dims):
        super(Attention, self).__init__()
        self.Wtemp = nn.Parameter(torch.randn(channels, spect_dims, 1))
        self.Wspect = nn.Parameter(torch.randn(channels, temp_dims, 1))
        self.Wtemp.requires_grad = True
        self.Wspect.requires_grad = True
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        fs = self.activation(torch.matmul(x, self.Wspect))  # fs = torch.matmul(x, self.Wspect)
        ft = self.activation(torch.matmul(x.permute(0, 1, 3, 2),
                                          self.Wtemp))  # ft = torch.matmul(x.permute(0,1,3,2), self.Wtemp)
        temp_scores = self.softmax(ft)
        spect_scores = self.softmax(fs)
        mask = torch.matmul(spect_scores, temp_scores.permute(0, 1, 3, 2)) * 100  #######
        x = x * mask
        return x


class ResBlockAttention(nn.Module):

    def __init__(self, inchannels, outchannels, spect_dims, temp_dims, kernelsize, stride, identity_downsample):
        super(ResBlockAttention, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernelsize, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.attention1 = Attention(outchannels, spect_dims, temp_dims)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernelsize, (1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.attention2 = Attention(outchannels, spect_dims, temp_dims)

    def forward(self, x):
        identity = x
        x = self.attention1(self.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.attention2(self.relu(x))
        return x


class CustomArch7(nn.Module):

    def __init__(self):
        super(CustomArch7, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module(
            'frontend',
            nn.Sequential(*[nn.Conv2d(1, 32, 3, padding="same"),
                            nn.BatchNorm2d(32), nn.ReLU()]))
        self.encoder.add_module('resblock1', ResBlockAttention(32, 32, 256, 32, 3, 1, None))
        self.flatten = nn.Flatten()
        inchannel = 32
        freq_dim = 256
        temp_dim = 32
        for i in range(5):
            freq_dim /= 2
            temp_dim /= 2
            indentity_downsample = nn.Conv2d(inchannel, 2 * inchannel, (1, 1), (2, 2))
            self.encoder.add_module(
                'resblock' + str(i + 2),
                ResBlockAttention(inchannel, 2 * inchannel, int(freq_dim), int(temp_dim), 3, 2, indentity_downsample))
            inchannel *= 2

    def forward(self, x):
        x = self.flatten(self.encoder(x))
        return torch.squeeze(x)


class ProjectionHead1(nn.Module):

    def __init__(self, d=128, h=1024 * 8, u=32):
        super(ProjectionHead1, self).__init__()
        assert h % d == 0, 'h must be divisible by d'
        v = h // d
        self.d = d
        self.h = h
        self.u = u
        self.v = v
        # print(f"d:{d}, h:{h}, u:{u}, v:{self.v}")
        self.linear1 = nn.Conv1d(d * v, d * u, kernel_size=(1, ), groups=d)
        self.elu = nn.ELU()
        self.linear2 = nn.Conv1d(d * u, d, kernel_size=(1, ), groups=d)

    def forward(self, x, norm=True):
        x = x.view(-1, self.h, 1)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        if norm:
            x = torch.nn.functional.normalize(x, p=2.0)

        return torch.squeeze(x, dim=-1)


class AttentionCNN(nn.Module):

    def __init__(self):
        super(AttentionCNN, self).__init__()

        self.encoder = CustomArch7()
        self.head = ProjectionHead1()
        self.full_model = nn.Sequential(self.encoder, self.head)

    def forward(self, x):
        return self.full_model(x)
