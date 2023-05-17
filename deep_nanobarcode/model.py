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
import torch.optim as optim
import numpy as np

from . import network_components as nc
from . import network_main as net_main
from . import dataset_handler as dat


def model_factory(nanobarcode_dataset, model_args, training_args):

    net = net_main.NanobarcodeClassifierNet(input_shape=dat.n_channels,
                                            output_shape=nanobarcode_dataset.n_proteins,
                                            **model_args).to(nc.nn_device)

    train_loader = torch.utils.data.DataLoader(nanobarcode_dataset.train_set,
                                               batch_size=training_args["batch_size"],
                                               shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(nanobarcode_dataset.val_set,
                                             batch_size=training_args["batch_size"],
                                             shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(nanobarcode_dataset.test_set,
                                              batch_size=training_args["batch_size"],
                                              shuffle=True, drop_last=False)

    optimizer = optim.AdamW(net.parameters(), lr=training_args["lr"], amsgrad=True)

    return net, train_loader, val_loader, test_loader, optimizer


def calc_metrics(net, data_loader, n_classes, full_output=False):

    confusion_matrix = np.zeros((n_classes, n_classes))

    net.eval()

    with torch.no_grad():

        for i, data in enumerate(data_loader):

            input_data, target_data = data

            outputs = torch.nn.functional.softmax(net(input_data), dim=1)

            predicted = torch.argmax(outputs, dim=1)

            for _t, _p in zip(target_data.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[_p, _t] += 1.0

    true_positive = []
    false_positive = []
    false_negative = []

    precision = []
    recall = []
    F1_score = []

    for n in range(n_classes):
        true_positive.append(confusion_matrix[n, n])
        false_positive.append(np.sum(confusion_matrix[n, :]) - confusion_matrix[n, n])
        false_negative.append(np.sum(confusion_matrix[:, n]) - confusion_matrix[n, n])

        precision.append(true_positive[-1] / max(true_positive[-1] + false_positive[-1], 1.0E-16))
        recall.append(true_positive[-1] / max(true_positive[-1] + false_negative[-1], 1.0E-16))
        F1_score.append(2.0 * precision[-1] * recall[-1] / max(precision[-1] + recall[-1], 1.0E-16))

    total_n_predictions = np.sum(confusion_matrix)

    overall_accuracy = np.sum(confusion_matrix.diagonal()) / total_n_predictions
    percent_false_positive = np.array(false_positive) / total_n_predictions
    percent_false_negative = np.array(false_negative) / total_n_predictions

    if full_output:

        return overall_accuracy, precision, recall, F1_score, percent_false_positive, percent_false_negative

    else:

        return overall_accuracy


def do_step(net, data_loader, loss_function, optimizer, do_train=True):

    running_loss = []

    if do_train:
        net.train()
    else:
        net.eval()

    for i, (input_data, target_data) in enumerate(data_loader, 0):

        if do_train:
            optimizer.zero_grad()

        outputs = net(input_data)

        _loss = loss_function(outputs, target_data)

        if do_train:
            _loss.backward()

            optimizer.step()

        running_loss.append(_loss.item())

    return np.mean(running_loss)


def separate_args(args):

    model_args = args.copy()

    training_args = dict()

    lr = model_args.pop("lr", None)
    batch_size = model_args.pop("batch_size", None)
    num_data_workers = model_args.pop("num_data_workers", None)

    training_args.update([('lr', lr), ('batch_size', batch_size), ('num_data_workers', num_data_workers)])

    return model_args, training_args


def hyperparameter_optim_objective(args, loss_function, n_proteins):

    net, train_loader, val_loader, test_loader, optimizer = model_factory(*separate_args(args))

    for epoch in range(5):
        do_step(net, train_loader, loss_function, optimizer, do_train=True)

        # test_loss = do_step(net, testloader, criterion, None, do_train=False)

    avg_val_accuracy = calc_metrics(net, val_loader, n_proteins, full_output=False)

    return avg_val_accuracy
