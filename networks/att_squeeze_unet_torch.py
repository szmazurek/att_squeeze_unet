import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FireModule(nn.Module):
    def __init__(self, fire_id, squeeze, expand, in_channels=3):
        super(FireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels, squeeze, kernel_size=1, padding="same"),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True),
        )
        self.left = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.right = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int, 1, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        filters,
        fire_id,
        squeeze,
        expand,
        strides,
        padding,
        deconv_ksize,
        att_filters,
        x_input_shape,
        g_input_shape,
    ):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels=x_input_shape[1],
            out_channels=filters,
            kernel_size=deconv_ksize,
            stride=strides,
            padding=padding,
            output_padding=0 if strides == (1, 1) else 1,
        )
        x_dummy = torch.zeros(x_input_shape)
        g_dummy = torch.zeros(g_input_shape)
        x_dummy_shape = self.upconv(x_dummy).shape
        self.attention = AttentionBlock(
            F_g=x_dummy_shape[1], F_l=g_input_shape[1], F_int=att_filters
        )
        g_dummy_shape = self.upconv_attention_block(x_dummy, g_dummy).shape
        self.fire = FireModule(
            fire_id, squeeze, expand, in_channels=g_dummy_shape[1]
        )

    def upconv_attention_block(self, x, g):
        d = self.upconv(x)
        x = self.attention(d, g)
        d = torch.cat([x, d], axis=1)
        return d

    def forward(self, x, g):
        d = self.upconv_attention_block(x, g)
        x = self.fire(d)
        return x


class AttSqueezeUNet(nn.Module):
    def __init__(self, n_classes, in_shape, dropout=False):
        super(AttSqueezeUNet, self).__init__()
        self._dropout = dropout
        x1_shape = [int(x / 2) for x in in_shape[-2:]]
        x2_shape = [int(x / 4) for x in in_shape[-2:]]
        x3_shape = [int(x / 8) for x in in_shape[-2:]]
        x4_shape = [int(x / 16) for x in in_shape[-2:]]
        padding_1 = self.calculate_same_padding(
            in_shape[-2:],
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.conv_1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=(2, 2), padding=padding_1
        )
        padding_2 = self.calculate_same_padding(
            x1_shape,
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.max_pooling_1 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=padding_2
        )
        self.fire_1 = FireModule(1, 16, 64, in_channels=64)
        self.fire_2 = FireModule(2, 16, 64, in_channels=128)
        padding_3 = self.calculate_same_padding(
            x2_shape,
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.max_pooling_2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=padding_3
        )

        self.fire_3 = FireModule(3, 32, 128, in_channels=128)
        self.fire_4 = FireModule(4, 32, 128, in_channels=256)
        padding_4 = self.calculate_same_padding(
            x3_shape,
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.max_pooling_3 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=padding_4
        )
        self.fire_5 = FireModule(5, 48, 192, in_channels=256)
        self.fire_6 = FireModule(6, 48, 192, in_channels=384)
        self.fire_7 = FireModule(7, 64, 256, in_channels=384)
        self.fire_8 = FireModule(8, 64, 256, in_channels=512)
        padding_5 = self.calculate_same_padding(
            x4_shape,
            kernel_size=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
        )
        self.upsampling_1 = UpsamplingBlock(
            filters=192,
            fire_id=9,
            squeeze=48,
            expand=192,
            padding=padding_5,
            strides=(1, 1),
            deconv_ksize=3,
            att_filters=96,
            x_input_shape=(1, 512, x4_shape[0], x4_shape[1]),
            g_input_shape=(1, 384, x4_shape[0], x4_shape[1]),
        )
        padding_6 = padding_5
        self.upsampling_2 = UpsamplingBlock(
            filters=128,
            fire_id=10,
            squeeze=32,
            expand=128,
            strides=(1, 1),
            deconv_ksize=3,
            att_filters=64,
            padding=padding_6,
            x_input_shape=(1, 384, x4_shape[0], x4_shape[1]),
            g_input_shape=(1, 256, x4_shape[0], x4_shape[1]),
        )
        padding_7 = self.calculate_same_padding(
            x4_shape, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)
        )
        self.upsampling_3 = UpsamplingBlock(
            filters=64,
            fire_id=11,
            squeeze=16,
            expand=64,
            padding=padding_7,
            strides=(2, 2),
            deconv_ksize=3,
            att_filters=16,
            x_input_shape=(1, 256, x4_shape[0], x4_shape[1]),
            g_input_shape=(1, 128, x3_shape[0], x3_shape[1]),
        )
        padding_8 = self.calculate_same_padding(
            x3_shape, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)
        )
        self.upsampling_4 = UpsamplingBlock(
            filters=32,
            fire_id=12,
            squeeze=16,
            expand=32,
            padding=padding_8,
            strides=(2, 2),
            deconv_ksize=3,
            att_filters=4,
            x_input_shape=(1, 128, x3_shape[0], x3_shape[1]),
            g_input_shape=(1, 64, x2_shape[0], x2_shape[1]),
        )

        self.upsampling_5 = nn.Upsample(scale_factor=2)
        self.upsampling_6 = nn.Upsample(scale_factor=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, padding="same"),
            nn.Softmax(dim=1) if n_classes > 1 else nn.Sigmoid(),
        )

    @staticmethod
    def calculate_same_padding(input_size, kernel_size, stride, dilation):
        if input_size[0] % stride[0] == 0:
            pad_along_height = max(
                dilation[0] * (kernel_size[0] - stride[0]), 0
            )
        else:
            pad_along_height = max(
                dilation[0] * (kernel_size[0] - (input_size[0] % stride[0])), 0
            )

        if input_size[1] % stride[1] == 0:
            pad_along_width = max(
                dilation[1] * (kernel_size[1] - stride[1]), 0
            )
        else:
            pad_along_width = max(
                dilation[1] * (kernel_size[1] - (input_size[1] % stride[1])), 0
            )

        p1 = math.ceil(pad_along_height / 2)
        p2 = math.ceil(pad_along_width / 2)
        return (p1, p2)

    @staticmethod
    def calculate_padding_equal(
        kernel_size, stride, input_size, output_size=None, dilation=(1, 1)
    ):
        """Calculate padding to keep the same input and output size."""

        if output_size is None:
            output_size = input_size  # assume same padding
        p1 = math.ceil(
            (
                (output_size[0] - 1) * stride[0]
                + 1
                + dilation[0] * (kernel_size[0] - 1)
                - input_size[0]
            )
            / 2
        )
        p2 = math.ceil(
            (
                (output_size[1] - 1) * stride[1]
                + 1
                + dilation[1] * (kernel_size[1] - 1)
                - input_size[1]
            )
            / 2
        )
        return (p1, p2)

    def forward(self, x):
        x0 = self.conv_1(x)
        x1 = self.max_pooling_1(x0)

        x2 = self.fire_1(x1)
        x2 = self.fire_2(x2)
        x2 = self.max_pooling_2(x2)

        x3 = self.fire_3(x2)
        x3 = self.fire_4(x3)
        x3 = self.max_pooling_3(x3)

        x4 = self.fire_5(x3)
        x4 = self.fire_6(x4)

        x5 = self.fire_7(x4)
        x5 = self.fire_8(x5)

        if self._dropout:
            x5 = F.dropout(x5, p=0.5)
        d5 = self.upsampling_1(x5, x4)
        d4 = self.upsampling_2(d5, x3)
        d3 = self.upsampling_3(d4, x2)
        d2 = self.upsampling_4(d3, x1)
        d1 = self.upsampling_5(d2)

        d0 = torch.cat([d1, x0], axis=1)
        d0 = self.conv_2(d0)
        d0 = self.upsampling_6(d0)
        d = self.conv_3(d0)

        return d
