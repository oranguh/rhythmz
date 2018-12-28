import logging
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def add_vgg_conv_block(index, layers, in_channels, out_channels, n_kernels, pool_size):
    layers["conv_{}".format(index)] = nn.Conv1d(
        in_channels, out_channels, n_kernels)
    layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
    index += 1

    layers["conv_{}".format(index)] = nn.Conv1d(
        out_channels, out_channels, n_kernels)
    layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
    layers["pool_{}".format(index)] = nn.MaxPool1d(pool_size)
    index += 1
    return index


class AudioClassifier(nn.Module):
    def __init__(self, combine, input_size, input_stride, n_classes):
        super().__init__()

        assert combine in {"MoT"}
        self.input_size = input_size
        self.n_classes = n_classes
        self.input_stride = input_stride
        self.combine = combine

        layers = OrderedDict()

        index = 1
        index = add_vgg_conv_block(index, layers, 1, 16, 9, 16)
        index = add_vgg_conv_block(index, layers, 16, 32, 3, 4)
        index = add_vgg_conv_block(index, layers, 32, 64, 3, 4)
        index = add_vgg_conv_block(index, layers, 64, 128, 3, 4)
        #index = add_vgg_conv_block(index, layers, 128, 256, 3, 4)

        self.layers = nn.Sequential(layers)

        with torch.no_grad():
            test_input = torch.zeros(1, 1, self.input_size)
            out = self.layers(test_input)
            self.feature_size = np.prod(out.size())

        log.info("Feature Size: {}".format(self.feature_size))

        clf_layers = OrderedDict()
        clf_layers["dropout_0"] = nn.Dropout(0.1)
        clf_layers["linear_1"] = nn.Linear(self.feature_size, 256)
        clf_layers["relu_1"] = nn.ReLU(inplace=True)
        clf_layers["dropout_1"] = nn.Dropout(0.1)
        clf_layers["linear_2"] = nn.Linear(256, self.n_classes)

        self.classifier = nn.Sequential(clf_layers)

        log.info("Created model : {}".format(self))

    def stack_strides(self, x):
        assert len(x.size()) == 1
        in_size = x.size(0)
        start = 0
        stacked = []
        # TODO: I'm using zero padding at the end - will this work?
        # TODO: how do I fix this?
        while start < in_size:
            split = x[start: (start+self.input_size)]
            if split.size(0) < self.input_size:
                zero_pad = torch.zeros(self.input_size - split.size(0))
                split = torch.cat((split, zero_pad))

            stacked.append(split)
            start += self.input_stride

        return torch.stack(stacked)

    def combine_stride_features(self, features):
        if self.combine == "MoT":
            feats, _ = features.max(0)
            return feats
        raise ValueError("Unavailable method '{}'".format(self.combine))

    def forward(self, x):
        # x is [number of input datapoints, size of datapoint]
        # assert len(x.size()) == 2

        out = torch.zeros(len(x), self.n_classes)
        # transform each to [batch, input_size] per datapoint
        # get the features, combine them
        # classify it!
        for idx, data_point in enumerate(x):
            stacked = self.stack_strides(data_point)
            features = self.layers(stacked.unsqueeze(1))
            features = features.squeeze()
            # flatten
            features = features.view(-1, self.feature_size)
            features = self.combine_stride_features(features)
            out[idx] = self.classifier(features.unsqueeze(0)).squeeze()

        # if not training, use a softmax
        if not self.training:
            out = F.softmax(out, 1)

        return out
