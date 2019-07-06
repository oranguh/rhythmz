import logging
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def add_vgg_conv_block(index, layers, in_channels, out_channels,
                       kernel_size, pool_size, stride=1,
                       batch_norm=False, direction="conv",
                       return_indices=False):
    if direction == "conv":
        layers["conv_{}".format(index)] = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride)
        if batch_norm:
            layers["batchnorm_{}".format(index)] = nn.BatchNorm1d(out_channels)
        layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
        index += 1

        layers["conv_{}".format(index)] = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=stride)
        if batch_norm:
            layers["batchnorm_{}".format(index)] = nn.BatchNorm1d(out_channels)
        layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
        layers["pool_{}".format(index)] = nn.MaxPool1d(
            pool_size, return_indices=return_indices)
        index += 1
        return index
    elif direction == "deconv":
        layers["pool_{}".format(index)] = nn.MaxUnpool1d(pool_size)

        layers["conv_{}".format(index)] = nn.ConvTranspose1d(
            out_channels, out_channels, kernel_size, stride=stride)
        if batch_norm:
            layers["batchnorm_{}".format(index)] = nn.BatchNorm1d(out_channels)
        layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
        index -= 1

        layers["conv_{}".format(index)] = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride)
        if batch_norm:
            layers["batchnorm_{}".format(index)] = nn.BatchNorm1d(out_channels)
        layers["relu_{}".format(index)] = nn.ReLU(inplace=True)
        index -= 1
        return index


def get_feature_layers(features, batch_norm):
    if features == "raw":
        layers = OrderedDict()

        index = 1
        index = add_vgg_conv_block(
            index, layers, 1, 16, 80, 16, stride=4, batch_norm=batch_norm)
        index = add_vgg_conv_block(
            index, layers, 16, 32, 3, 4, batch_norm=batch_norm)
        index = add_vgg_conv_block(
            index, layers, 32, 64, 3, 4, batch_norm=batch_norm)
        index = add_vgg_conv_block(
            index, layers, 64, 32, 2, 4, batch_norm=batch_norm)

    elif features == "ms":
        layers = OrderedDict()

        layers["conv_1"] = nn.Conv2d(
            1, 16, 7, stride=1)
        layers["relu_1"] = nn.ReLU(inplace=True)
        if batch_norm:
            layers["batchnorm_1"] = nn.BatchNorm2d(16)
        layers["pool_1"] = nn.MaxPool2d(3)

        layers["conv_2"] = nn.Conv2d(
            16, 32, 5, stride=1)
        layers["relu_2"] = nn.ReLU(inplace=True)
        if batch_norm:
            layers["batchnorm_2"] = nn.BatchNorm2d(32)
        layers["pool_2"] = nn.MaxPool2d(3)

        layers["conv_3"] = nn.Conv2d(
            32, 64, 3, stride=1)
        layers["relu_3"] = nn.ReLU(inplace=True)
        if batch_norm:
            layers["batchnorm_3"] = nn.BatchNorm2d(64)
        layers["pool_3"] = nn.MaxPool2d(3)

        layers["conv_4"] = nn.Conv2d(
            64, 128, 3, stride=1)
        layers["relu_4"] = nn.ReLU(inplace=True)
        if batch_norm:
            layers["batchnorm_4"] = nn.BatchNorm2d(128)
        layers["pool_4"] = nn.MaxPool2d(3)

    return layers


class LibrivoxAudioClassifier(nn.Module):
    def __init__(self, features, n_classes, device, batch_norm=False):
        super().__init__()

        assert features in {"ms", "raw"}
        self.n_classes = n_classes
        self.features = features
        self.device = device

        layers = get_feature_layers(self.features, batch_norm)

        self.layers = nn.Sequential(layers)
        self.feature_size = 128

        log.info("Feature Size: {}".format(self.feature_size))

        self.classifier = nn.Sequential(self.get_classifier_layers())

        log.info("Created model : {}".format(self))

    def get_classifier_layers(self):
        clf_layers = OrderedDict()
        clf_layers["dropout_0"] = nn.Dropout(0.3)
        clf_layers["linear_1"] = nn.Linear(self.feature_size, 256)
        clf_layers["relu_1"] = nn.ReLU(inplace=True)
        clf_layers["dropout_1"] = nn.Dropout(0.3)
        clf_layers["linear_2"] = nn.Linear(256, self.n_classes)

        return clf_layers

    def get_features(self, x):
        return self.layers(x).view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self.get_features(x))


class LibrivoxAudioClassifier2(LibrivoxAudioClassifier):
    def __init__(self, features, n_classes, device, batch_norm=False):
        super().__init__(features, n_classes, device, batch_norm)

        aux_clf_layers = OrderedDict()

        # this is the auxiliary classifier which learns whether
        # the two clips comes from the same speaker, based on the features
        aux_clf_layers["aux_dropout"] = nn.Dropout(0.3)
        aux_clf_layers["aux_linear"] = nn.Linear(self.feature_size, 1)

        self.aux_clf = nn.Sequential(aux_clf_layers)

    def forward2(self, x_1, x_2):
        f_1 = self.get_features(x_1)
        f_2 = self.get_features(x_2)

        assert f_1.size() == f_2.size()
        return self.aux_clf(f_1 - f_2).view(x_1.size(0))


class LibrivoxAudioEncDec(nn.Module):
    def __init__(self, features, n_classes, device, batch_norm=False):
        super().__init__()

        assert features in {"ms", "raw"}
        self.n_classes = n_classes
        self.features = features
        self.device = device
        self.batch_norm = batch_norm

        encoder_layers, decoder_layers = self.get_layers()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.feature_size = 128

        log.info("Feature Size: {}".format(self.feature_size))

        clf_layers = OrderedDict()
        clf_layers["dropout_0"] = nn.Dropout(0.3)
        clf_layers["linear_1"] = nn.Linear(self.feature_size, 256)
        clf_layers["relu_1"] = nn.ReLU(inplace=True)
        clf_layers["dropout_1"] = nn.Dropout(0.3)
        clf_layers["linear_2"] = nn.Linear(256, self.n_classes)

        self.classifier = nn.Sequential(clf_layers)

        log.info("Created model : {}".format(self))

    def _get_raw_layer(self, direction, in_channels, out_channels, kernel_size, pool_size, stride):
        layers = []
        if direction == "conv":
            layers.append(nn.Conv1d(in_channels, out_channels,
                                    kernel_size, stride=stride))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv1d(out_channels, out_channels,
                                    kernel_size, stride=stride))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(pool_size, return_indices=True))
        else:
            layers.append(nn.MaxUnpool1d(pool_size))
            layers.append(nn.ConvTranspose1d(
                out_channels, out_channels, kernel_size, stride=stride))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.ConvTranspose1d(
                out_channels, in_channels, kernel_size, stride=stride))
            layers.append(nn.ReLU(True))

        return layers

    def _get_ms_layer(self, direction, in_channels, out_channels, kernel_size, pool_size, stride):
        if direction == "conv":
            return (
                nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride=stride),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size, return_indices=True)
            )

        elif direction == "deconv":
            return (
                nn.MaxUnpool2d(pool_size),
                nn.ConvTranspose2d(out_channels, in_channels,
                                   kernel_size, stride=stride),
                nn.ReLU(True)
            )

    def get_layers(self):
        batch_norm = self.batch_norm
        if self.features == "raw":
            encoder_layers = []

            encoder_layers.extend(self._get_raw_layer("conv",
                                                      in_channels=1,
                                                      out_channels=32,
                                                      kernel_size=7,
                                                      pool_size=15,
                                                      stride=4))

            decoder_layers = []

            decoder_layers.extend(self._get_raw_layer("deconv",
                                                      in_channels=1,
                                                      out_channels=32,
                                                      kernel_size=7,
                                                      pool_size=15,
                                                      stride=4))

            return encoder_layers, decoder_layers
        else:
            encoder_layers = []

            # layers["conv_1"] = nn.Conv2d(
            # 1, 16, 7, stride=1)
            # layers["relu_1"] = nn.ReLU(inplace=True)
            # if batch_norm:
            #     layers["batchnorm_1"] = nn.BatchNorm2d(16)
            # layers["pool_1"] = nn.MaxPool2d(3)
            encoder_layers.extend(self._get_ms_layer(
                "conv", in_channels=1,
                out_channels=16, kernel_size=7,
                pool_size=3, stride=1
            ))
            # layers["conv_2"] = nn.Conv2d(
            #     16, 32, 5, stride=1)
            # layers["relu_2"] = nn.ReLU(inplace=True)
            # if batch_norm:
            #     layers["batchnorm_2"] = nn.BatchNorm2d(32)
            # layers["pool_2"] = nn.MaxPool2d(3)
            encoder_layers.extend(self._get_ms_layer(
                "conv", in_channels=16,
                out_channels=32, kernel_size=5,
                pool_size=3, stride=1
            ))
            # layers["conv_3"] = nn.Conv2d(
            #     32, 64, 3, stride=1)
            # layers["relu_3"] = nn.ReLU(inplace=True)
            # if batch_norm:
            #     layers["batchnorm_3"] = nn.BatchNorm2d(64)
            # layers["pool_3"] = nn.MaxPool2d(3)
            encoder_layers.extend(self._get_ms_layer(
                "conv", in_channels=32,
                out_channels=64, kernel_size=3,
                pool_size=3, stride=1
            ))

            # layers["conv_4"] = nn.Conv2d(
            #     64, 128, 3, stride=1)
            # layers["relu_4"] = nn.ReLU(inplace=True)
            # if batch_norm:
            #     layers["batchnorm_4"] = nn.BatchNorm2d(128)
            # layers["pool_4"] = nn.MaxPool2d(3)
            encoder_layers.extend(self._get_ms_layer(
                "conv", in_channels=64,
                out_channels=128, kernel_size=3,
                pool_size=3, stride=1
            ))

            decoder_layers = []

            # decoder_layers.extend((
            #     nn.ConvTranspose2d(128, 64, 3, 1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxUnpool2d(3),
            #     nn.ConvTranspose2d(64, 32, 3, 1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxUnpool2d(3),
            #     nn.ConvTranspose2d(32, 16, 3, 1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxUnpool2d(3)
            # ))

            decoder_layers.extend(self._get_ms_layer(
                "deconv", in_channels=64,
                out_channels=128, kernel_size=3,
                pool_size=3, stride=1
            ))

            # remove the first un pooling layer
            del decoder_layers[0]

            decoder_layers.extend(self._get_ms_layer(
                "deconv", in_channels=32,
                out_channels=64, kernel_size=3,
                pool_size=3, stride=1
            ))

            decoder_layers.extend(self._get_ms_layer(
                "deconv", in_channels=16,
                out_channels=32, kernel_size=5,
                pool_size=3, stride=1
            ))

            decoder_layers.extend(self._get_ms_layer(
                "deconv", in_channels=1,
                out_channels=16, kernel_size=7,
                pool_size=3, stride=1
            ))

            del decoder_layers[-1]

            assert not isinstance(decoder_layers[-1], nn.ReLU)

            return encoder_layers, decoder_layers

    def forward(self, x_1, x_2):
        assert x_1.size(0) == x_2.size(0)

        max_pool_indices_size = {}

        out = x_1
        for module in self.encoder_layers:
            inp_size = out.size()
            print("\t", module, "Input: {}, ".format(out.size()))
            if isinstance(module, nn.MaxPool1d) or isinstance(module, nn.MaxPool2d):
                out, indices = module(out)
                print("\t", len(max_pool_indices_size), inp_size)
                max_pool_indices_size[len(max_pool_indices_size)] = (
                    indices, inp_size)
            else:
                out = module(out)
            print("\t\tOutput: {}".format(out.size()))

        # out = out.view(x_1.size(0), -1)
        print()
        # decoding layer
        # -2 because we remove one MaxUnpool
        mp_idx = len(max_pool_indices_size) - 2
        dec_out = out
        for module in self.decoder_layers:
            print("\t", module, "Input: {}, ".format(dec_out.size()), end="")
            if isinstance(module, nn.MaxUnpool1d) or isinstance(module, nn.MaxUnpool2d):
                indices, size = max_pool_indices_size[mp_idx]
                dec_out = module(dec_out, indices, output_size=size)
                mp_idx -= 1
            else:
                dec_out = module(dec_out)
            print("Output: {}".format(dec_out.size()))

        # dec_out = out
        # dec_out = self.decoder_layers[0](dec_out)
        # dec_out = self.decoder_layers[1](dec_out)
        # indices, size = max_pool_indices_size[2]
        # dec_out = self.decoder_layers[2](dec_out, indices, size)

        print(dec_out.size())
