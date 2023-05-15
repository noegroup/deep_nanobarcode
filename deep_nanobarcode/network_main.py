# The MIT License (MIT)
#
# Copyright (c) 2022-2023, Mohsen Sadeghi (mohsen.sadeghi@fu-berlin)
# Artificial Intelligence for the Sciences Group (AI4Science),
# Freie Universität Berlin, Germany.
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from . import dataset_handler as dat
from . import network_components as nc
import numpy as np


class NanobarcodeClassifierNet(nn.Module):

    def __init__(self, input_shape=dat.n_channels, output_shape=None,
                 width=20, n_middle_layers=12, use_resnet=True, resnet_stride=2,
                 cardinality=5, use_dropout=True, use_bn=True, dropout_rate=0.4):

        super(NanobarcodeClassifierNet, self).__init__()

        activation_args = {'use_dropout': use_dropout,
                           'use_bn': use_bn,
                           'dropout_rate': dropout_rate}

        self.resnet_stride = resnet_stride

        #         self.batch_norm_0 = nn.BatchNorm1d(num_features=input_shape)

        self.initial_layers = nn.ModuleList()

        mid_width = (input_shape + width) // 2

        self.initial_layers.append(nn.Linear(input_shape, mid_width, bias=True))
        self.initial_layers.append(nc.CompoundActivationBlock(use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                              use_bn=use_bn, num_features=mid_width))

        self.initial_layers.append(nn.Linear(mid_width, width, bias=True))
        self.initial_layers.append(nc.CompoundActivationBlock(use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                              use_bn=use_bn, num_features=width))

        self.middle_layers = nn.ModuleList()

        for j in range(cardinality):

            branch = nn.ModuleList()

            if use_resnet:

                for i in range(n_middle_layers):
                    branch.append(nc.ResidualBlock(block_depth=resnet_stride, num_features=width,
                                                   use_bias=False,
                                                   **activation_args))
            else:

                for i in range(n_middle_layers):
                    branch.append(nn.Linear(width, width, bias=False))
                    branch.append(nc.CompoundActivationBlock(use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                             use_bn=use_bn, num_features=width))

            self.middle_layers.append(branch)

        mid_width = (output_shape + width) // 2

        self.final_layers = nn.ModuleList()

        self.final_layers.append(nn.Linear(width, mid_width, bias=True))
        self.final_layers.append(nc.CompoundActivationBlock(use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                            use_bn=use_bn, num_features=mid_width))
        self.final_layers.append(nn.Linear(mid_width, output_shape, bias=True))
        self.final_layers.append(nc.CompoundActivationBlock(use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                            use_bn=False, num_features=output_shape))
        self.final_layers.append(nn.Linear(output_shape, output_shape, bias=True))

        self.identity = nn.Identity()

    def forward(self, x, **kwargs):

        #         x = self.batch_norm_0(x)

        for _lay in self.initial_layers:
            x = _lay(x, **kwargs)

        y = self.identity(x)

        for block in self.middle_layers[0]:
            y = block(y, **kwargs)

        z = self.identity(y)

        for branch in self.middle_layers[1:]:

            y = self.identity(x)

            for block in branch:
                y = block(y, **kwargs)

            z = torch.add(z, y)

        for _lay in self.final_layers:
            z = _lay(z, **kwargs)

        return z


class ContrastModifier(nn.Module):

    def __init__(self):
        super(ContrastModifier, self).__init__()

        self.mu = nn.Parameter(torch.zeros(dat.n_channels, dtype=torch.float32, device=nc.nn_device), requires_grad=True)
        self.sig = nn.Parameter(torch.ones(dat.n_channels, dtype=torch.float32, device=nc.nn_device), requires_grad=True)

    def forward(self, x):
        return self.sig * (x - self.mu)


def feed_to_network(net, image_slice_scaled):

    normalizing_layer = nn.Softmax(dim=-1)

    net.eval()

    with torch.no_grad():

        _im = image_slice_scaled.copy().reshape(dat.n_channels, -1).transpose().astype(np.float32)

        input_data = torch.from_numpy(_im).float().to(nc.nn_device)
        raw_output = net(input_data)
        predicted = normalizing_layer(raw_output).cpu().numpy()

        entropy = np.mean(np.sum(-predicted * np.log(predicted + 1.0e-16), axis=1))

    return predicted, entropy


def feed_to_network_and_optimize_scaling(net, image_slice_scaled, n_optim_iter):

    _im = image_slice_scaled.copy().reshape(dat.n_channels, -1).transpose().astype(np.float32)

    normalizing_layer = nn.Softmax(dim=-1)

    uber_net = ContrastModifier().to(nc.nn_device)
    uber_optimizer = torch.optim.AdamW(uber_net.parameters(), lr=0.001, amsgrad=True)

    uber_net.train()

    net.eval()

    for _param in net.parameters():
        _param.requires_grad = False

    input_data = torch.from_numpy(_im).float().to(nc.nn_device)

    entropy_list = []

    for i in range(n_optim_iter):
        uber_optimizer.zero_grad()

        raw_output = net(uber_net(input_data))
        predicted = normalizing_layer(raw_output)

        entropy = torch.mean(torch.sum(-predicted * torch.log(predicted + 1.0e-16), dim=1))

        entropy.backward()

        uber_optimizer.step()

        entropy_list.append(entropy.detach().cpu().numpy())

    uber_net.eval()

    with torch.no_grad():

        raw_output = net(uber_net(input_data))

        predicted = normalizing_layer(raw_output)

        entropy = torch.mean(torch.sum(-predicted * torch.log(predicted + 1.0e-16), dim=1))

    return predicted.cpu().numpy(), entropy.cpu().numpy(), entropy_list
