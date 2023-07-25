import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPool2D,
    BatchNormalization,
    ReLU,
    LeakyReLU,
    UpSampling2D,
    Activation,
    ZeroPadding2D,
    Lambda,
    AveragePooling2D,
    Reshape,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model, Sequential


# class AttSqueezeUNetTorch(nn.Module):
class FireModuleTF(tf.keras.Model):
    def __init__(self, fire_id, squeeze, expand):
        super(FireModuleTF, self).__init__(name="")

        self.fire = Sequential()
        self.fire.add(
            Conv2D(
                squeeze,
                (1, 1),
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            )
        )
        self.fire.add(BatchNormalization(axis=-1))
        self.left = Conv2D(
            expand,
            (1, 1),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        self.right = Conv2D(
            expand,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )

    def call(self, x):
        x = self.fire(x)
        print(f"tf fire {x.shape}")
        left = self.left(x)
        print(f"tf left {left.shape}")
        right = self.right(x)
        print(f"tf right {right.shape}")
        x = tf.concat([left, right], axis=-1)
        print(f"tf out {x.shape}")
        return x


class FireModule(nn.Module):
    def __init__(self, fire_id, squeeze, expand):
        super(FireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(3, squeeze, kernel_size=1, padding="same"),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True),
        )
        self.left = nn.Conv2d(squeeze, expand, kernel_size=1, padding="same")
        self.right = nn.Conv2d(squeeze, expand, kernel_size=3, padding="same")

    def forward(self, x):
        x = self.fire(x)
        print(f"torch fire {x.shape}")
        left = self.left(x)
        print(f"torch left {left.shape}")
        right = self.right(x)
        print(f"torch right {right.shape}")
        x = torch.cat([left, right], dim=1)
        print(f"torch out {x.shape}")
        return x
