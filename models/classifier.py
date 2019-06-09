import logging
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def add_vgg_conv_block(index, layers, in_channels, out_channels, kernel_size, pool_size, stride=1):
    layers["conv_{}".format(index)] = nn.Conv1d(
        in_channels, out_channels, kernel_size, stride=stride)
    # layers["batchnorm_{}".format(index)] = nn.BatchNorm1d(out_channels)
    layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
    index += 1

    layers["conv_{}".format(index)] = nn.Conv1d(
        out_channels, out_channels, kernel_size, stride=stride)
    # layers["batchnorm_{}".format(index)] = nn.BatchNorm1d(out_channels)
    layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
    layers["pool_{}".format(index)] = nn.MaxPool1d(pool_size)
    index += 1
    return index


def get_feature_layers(features):
    if features == "raw":
        layers = OrderedDict()

        index = 1
        index = add_vgg_conv_block(index, layers, 1, 16, 80, 16, stride=4)
        index = add_vgg_conv_block(index, layers, 16, 32, 3, 4)
        index = add_vgg_conv_block(index, layers, 32, 64, 3, 4)

    elif features == "mel-spectogram":
        layers = OrderedDict()

        layers["conv_1"] = nn.Conv2d(
            1, 16, 7, stride=1)
        layers["relu_1"] = nn.ReLU(inplace=True)
        # layers["batchnorm_1"] = nn.BatchNorm2d(16)
        layers["pool_1"] = nn.MaxPool2d(3)

        layers["conv_2"] = nn.Conv2d(
            16, 32, 5, stride=1)
        layers["relu_2"] = nn.ReLU(inplace=True)
        # layers["batchnorm_2"] = nn.BatchNorm2d(32)
        layers["pool_2"] = nn.MaxPool2d(3)

        layers["conv_3"] = nn.Conv2d(
            32, 64, 3, stride=1)
        layers["relu_3"] = nn.ReLU(inplace=True)
        # layers["batchnorm_3"] = nn.BatchNorm2d(64)
        layers["pool_3"] = nn.MaxPool2d(3)

        layers["conv_4"] = nn.Conv2d(
            64, 128, 3, stride=1)
        layers["relu_4"] = nn.ReLU(inplace=True)
        # layers["batchnorm_4"] = nn.BatchNorm2d(128)
        layers["pool_4"] = nn.MaxPool2d(3)

    return layers


class LibrivoxAudioClassifier(nn.Module):
    def __init__(self, features, n_classes, device):
        super().__init__()

        assert features in {"mel-spectogram", "raw"}
        self.n_classes = n_classes
        self.features = features
        self.device = device

        layers = get_feature_layers(self.features)

        self.layers = nn.Sequential(layers)
        self.feature_size = 512

        log.info("Feature Size: {}".format(self.feature_size))

        clf_layers = OrderedDict()
        clf_layers["dropout_0"] = nn.Dropout(0.3)
        clf_layers["linear_1"] = nn.Linear(self.feature_size, 256)
        clf_layers["relu_1"] = nn.ReLU(inplace=True)
        clf_layers["dropout_1"] = nn.Dropout(0.3)
        clf_layers["linear_2"] = nn.Linear(256, self.n_classes)

        self.classifier = nn.Sequential(clf_layers)

        log.info("Created model : {}".format(self))

    def forward(self, x):
        features = self.layers(x)
        return self.classifier(features)
