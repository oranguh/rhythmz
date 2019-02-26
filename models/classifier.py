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


def get_feature_layers(features, input_size, input_stride):
    if features == "raw":
        layers = OrderedDict()

        index = 1
        index = add_vgg_conv_block(index, layers, 1, 16, 80, 16, stride=4)
        index = add_vgg_conv_block(index, layers, 16, 32, 3, 4)
        index = add_vgg_conv_block(index, layers, 32, 64, 3, 4)
        # index = add_vgg_conv_block(index, layers, 64, 128, 3, 4)
        # index = add_vgg_conv_block(index, layers, 128, 256, 3, 4)

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


class AudioClassifier(nn.Module):
    def __init__(self, combine, features, input_size, input_stride, n_classes, device):
        super().__init__()

        assert combine in {"MoT", "LSTM"}
        assert features in {"mel-spectogram", "raw"}

        self.input_size = input_size
        self.n_classes = n_classes
        self.input_stride = input_stride
        self.combine = combine
        self.features = features
        self.device = device

        layers = get_feature_layers(
            features, self.input_size, self.input_stride)

        self.layers = nn.Sequential(layers)
        with torch.no_grad():
            if self.features == "raw":
                test_input = torch.zeros(1, 1, self.input_size)
            elif self.features == "mel-spectogram":
                test_input = torch.zeros(
                    1, 1, 128, self.input_size)
            out = self.layers(test_input)
            self.feature_size = int(np.prod(out.size()))

        log.info("Feature Size: {}".format(self.feature_size))

        if self.combine == "LSTM":
            # initialize the LSTM
            self.lstm_hidden_size = self.feature_size
            self.lstm_num_layers = 1
            self.lstm = nn.LSTM(input_size=self.feature_size,
                                hidden_size=self.feature_size, num_layers=self.lstm_num_layers)

        clf_layers = OrderedDict()
        clf_layers["dropout_0"] = nn.Dropout(0.3)
        clf_layers["linear_1"] = nn.Linear(self.feature_size, 256)
        clf_layers["relu_1"] = nn.ReLU(inplace=True)
        clf_layers["dropout_1"] = nn.Dropout(0.3)
        clf_layers["linear_2"] = nn.Linear(256, self.n_classes)

        self.classifier = nn.Sequential(clf_layers)

        log.info("Created model : {}".format(self))

    def stack_strides(self, x):
        if self.features == "raw":
            in_size = x.size(0)
            start = 0
            stacked = []
            # TODO: I'm using zero padding at the end - will this work?
            # TODO: how do I fix this?
            while start < in_size:
                split = x[start: (start+self.input_size)]
                if split.size(0) < self.input_size:
                    zero_pad = torch.zeros(
                        self.input_size - split.size(0)).to(self.device)
                    split = torch.cat((split, zero_pad))

                stacked.append(split)
                start += self.input_stride

            return torch.stack(stacked)
        else:
            stacked = []
            in_size = x.size(1)
            start = 0
            stacked = []
            while start < in_size:
                split = x[:, start: (start + self.input_size)]
                if split.size(1) < self.input_size:
                    zero_pad = torch.zeros((x.size(0),
                                            self.input_size - split.size(1))).to(self.device)
                    split = torch.cat((split, zero_pad), dim=1)
                stacked.append(split)
                start += self.input_stride

            return torch.stack(stacked)

    def _init_hidden(self):
        return (torch.randn(self.lstm_num_layers, 1, self.lstm_hidden_size).to(self.device),
                torch.randn(self.lstm_num_layers, 1, self.lstm_hidden_size).to(self.device))

    def combine_stride_features(self, features):
        if self.combine == "MoT":
            feats, _ = features.max(0)
            return feats
        elif self.combine == "LSTM":
            hidden = self._init_hidden()
            N = features.size(0)
            lstm_out, hidden = self.lstm(features.view(N, 1, -1), hidden)
            return lstm_out[-1]
        raise ValueError("Unavailable method '{}'".format(self.combine))
    def marcos_combine_stride_features(self, features, stacked_batches_sizes):
        if self.combine == "MoT":
            feats_batched = []
            start_idx = 0
            for batch_sizer in stacked_batches_sizes:
                batch_sizer = start_idx + batch_sizer
                feats, _ = features[start_idx:batch_sizer].max(0)
                feats_batched.append(feats)
                # print(batch_sizer)
                start_idx = batch_sizer

            return feats_batched
        elif self.combine == "LSTM":
            feats_batched = []
            start_idx = 0
            for batch_sizer in stacked_batches_sizes:
                batch_sizer = start_idx + batch_sizer
                hidden = self._init_hidden()
                N = features[start_idx:batch_sizer].size(0)
                print(features[start_idx:batch_sizer].view(N, 1, -1).shape)
                print(features[start_idx:batch_sizer].size(0), features[0].shape)
                lstm_out, hidden = self.lstm(features[start_idx:batch_sizer].view(N, 1, -1), hidden)
                start_idx = batch_sizer
                feats_batched.append(lstm_out[-1])
            print(feats_batched[0].shape)
            print(a)
            return feats_batched
        raise ValueError("Unavailable method '{}'".format(self.combine))

    def forward(self, x):
        out = torch.zeros(len(x), self.n_classes).to(self.device)
        # transform each to [batch, input_size] per datapoint
        # get the features, combine them
        # classify it!

        if True:
            batched_features = torch.zeros(len(x), self.feature_size).to(self.device)
            stacked_batched_data = []
            stacked_batches_sizes = []

            for idx, data_point in enumerate(x):
                stacked = self.stack_strides(data_point).unsqueeze(1)
                stacked_batched_data.append(stacked)
                stacked_batches_sizes.append(stacked.size(0))

            stacked_batch = torch.cat(stacked_batched_data, dim=0)
            features = self.layers(stacked_batch)
            features = features.squeeze()
            # flatten
            features = self.marcos_combine_stride_features(features, stacked_batches_sizes)
            batched_features = torch.stack(features)
            out = self.classifier(batched_features.view(len(x), self.feature_size))

        if False:
            batched_features = torch.zeros(len(x), self.feature_size).to(self.device)
            for idx, data_point in enumerate(x):
                stacked = self.stack_strides(data_point).unsqueeze(1)
                # print(stacked.shape)
                features = self.layers(stacked)
                # features = self.layers(data_point.unsqueeze(0).unsqueeze(0))
                features = features.squeeze()
                # flatten
                features = features.view(stacked.size(0), self.feature_size)
                features = self.combine_stride_features(features)
                # print(features.unsqueeze(0).shape)
                batched_features[idx] = features.unsqueeze(0)

            out = self.classifier(batched_features)


        if not self.training:
            out = F.softmax(out, 1)

        return out
