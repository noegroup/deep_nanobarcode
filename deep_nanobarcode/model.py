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
import matplotlib.pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import tifffile

from . import network_components
from . import network_main
from . import dataset_handler
from . import image_processing


def model_factory(nanobarcode_dataset, model_args, training_args):

    net = network_main.NanobarcodeClassifierNet(input_shape=dataset_handler.n_channels,
                                                output_shape=nanobarcode_dataset.n_proteins,
                                                **model_args).to(network_components.nn_device)

    # torch.multiprocessing.set_start_method("spawn")

    extra_kwarg = {"shuffle": True, "drop_last": False}
    # {"num_workers": training_args["num_data_workers"]}

    train_data_loader = torch.utils.data.DataLoader(nanobarcode_dataset.train_set,
                                                    batch_size=training_args["batch_size"],
                                                    **extra_kwarg)

    val_data_loader = torch.utils.data.DataLoader(nanobarcode_dataset.val_set,
                                                  batch_size=training_args["batch_size"])

    test_data_loader = torch.utils.data.DataLoader(nanobarcode_dataset.test_set,
                                                   batch_size=training_args["batch_size"])

    optimizer = optim.AdamW(net.parameters(), lr=training_args["lr"], amsgrad=True)

    return net, train_data_loader, val_data_loader, test_data_loader, optimizer


def calc_metrics(net, data_loader, verbose=False) -> dict:

    n_classes = 0

    for i, data in enumerate(data_loader):
        input_data, target_data = data
        n_classes = max(n_classes, target_data.cpu().numpy().max() + 1)

    if verbose:
        print(f"Existence of {n_classes} classes was inferred from the data.")

    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)

    net.eval()

    with torch.no_grad():

        for i, data in enumerate(data_loader):

            input_data, target_data = data

            outputs = torch.nn.functional.softmax(net(input_data), dim=1)

            predicted = torch.argmax(outputs, dim=1)

            for _t, _p in zip(target_data.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[_p, _t] += 1.0

    for i in range(confusion_matrix.shape[1]):
        confusion_matrix[:, i] /= np.sum(confusion_matrix[:, i])

    metric = {"true positive": [], "false positive": [], "false negative": [],
              "precision": [], "recall": [], "F1-score": []}

    for n in range(n_classes):

        _true_positive = confusion_matrix[n, n]
        _false_positive = np.sum(confusion_matrix[n, :]) - confusion_matrix[n, n]
        _false_negative = np.sum(confusion_matrix[:, n]) - confusion_matrix[n, n]

        _precision = _true_positive / max(_true_positive + _false_positive, 1.0E-16)
        _recall = _true_positive / max(_true_positive + _false_negative, 1.0E-16)
        _F1_score = 2.0 * _precision * _recall / max(_precision + _recall, 1.0E-16)

        metric["true positive"].append(_true_positive)
        metric["false positive"].append(_false_positive)
        metric["false negative"].append(_false_negative)

        metric["precision"].append(_precision)
        metric["recall"].append(_recall)
        metric["F1-score"].append(_F1_score)

    total_n_predictions = np.sum(confusion_matrix)

    metric["overall accuracy"] = np.sum(confusion_matrix.diagonal()) / total_n_predictions
    metric["percent false positive"] = np.array(metric["false positive"]) / total_n_predictions
    metric["percent false negative"] = np.array(metric["false negative"]) / total_n_predictions

    return metric


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


def training_loop(net, train_data_loader, val_data_loader, optimizer,
                  num_epochs=120, starting_epoch=0, allow_overfit_steps=10,
                  losses=None, accuracies=None,
                  save_checkpoint=True, starting_max_accuracy=0.0,
                  save_net_file_name="../network_params/weights_generic.pth"):

    loss_function = nn.CrossEntropyLoss()

    lr0 = 0.0005

    for param_group in optimizer.param_groups:
        lr0 = param_group['lr']

    lr_drop = 0.9
    epochs_drop = 20

    epoch = starting_epoch
    overfit = 0

    max_accuracy = starting_max_accuracy

    if losses is None:
        losses = {"training": [], "validation": [], "test": []}

    if accuracies is None:
        accuracies = {"training": [], "validation": [], "test": []}

    while overfit < allow_overfit_steps and epoch < num_epochs:

        epoch += 1

        train_loss = do_step(net, train_data_loader, loss_function, optimizer, do_train=True)
        val_loss = do_step(net, val_data_loader, loss_function, None, do_train=False)

        if train_loss < 0.8 * val_loss:
            overfit += 1
            print("overfit alert!")

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr0 * lr_drop ** (np.floor((1.0 + float(epoch)) / epochs_drop))
            print(f"lr = {param_group['lr']}")

        train_metric = calc_metrics(net, train_data_loader)
        val_metric = calc_metrics(net, val_data_loader)

        if save_checkpoint and val_metric["overall accuracy"] > max_accuracy:
            print("saved as the best version so far...")
            torch.save(net.state_dict(), save_net_file_name)
            max_accuracy = val_metric["overall accuracy"]

        losses["training"].append(train_loss)
        losses["validation"].append(val_loss)

        accuracies["training"].append(train_metric["overall accuracy"])
        accuracies["validation"].append(val_metric["overall accuracy"])

        print("epoch: {:d} -- train loss : {:7.5f} -- validation loss : {:7.5f} -- validation accuracy : {:4.2f}%"
              .format(epoch, train_loss, val_loss, val_metric["overall accuracy"] * 100.0))

    return losses, accuracies, epoch


def separate_args(args):

    model_args = args.copy()

    training_args = dict()

    lr = model_args.pop("lr", None)
    batch_size = model_args.pop("batch_size", None)
    num_data_workers = model_args.pop("num_data_workers", None)

    training_args.update([('lr', lr), ('batch_size', batch_size), ('num_data_workers', num_data_workers)])

    return model_args, training_args


def hyperparameter_optim_objective(args, loss_function):

    net, train_loader, val_loader, test_loader, optimizer = model_factory(*separate_args(args))

    for epoch in range(5):
        do_step(net, train_loader, loss_function, optimizer, do_train=True)

        # test_loss = do_step(net, testloader, criterion, None, do_train=False)

    result = calc_metrics(net, val_loader)

    return result["overall accuracy"]


def feed_to_network(net, image_slice_scaled):

    normalizing_layer = nn.Softmax(dim=-1)

    net.eval()

    with torch.no_grad():

        _im = image_slice_scaled.copy().reshape(dataset_handler.n_channels, -1).transpose().astype(np.float32)

        input_data = torch.from_numpy(_im).float().to(network_components.nn_device)
        raw_output = net(input_data)
        predicted = normalizing_layer(raw_output).cpu().numpy()

        entropy = np.mean(np.sum(-predicted * np.log(predicted + 1.0e-16), axis=1))

    return predicted, entropy


def feed_to_network_and_optimize_channel_scaling(net, image_slice_scaled, n_optim_iter):

    _im = image_slice_scaled.copy().reshape(dataset_handler.n_channels, -1).transpose().astype(np.float32)

    normalizing_layer = nn.Softmax(dim=-1)

    uber_net = network_main.ContrastModifier().to(network_components.nn_device)
    uber_optimizer = torch.optim.AdamW(uber_net.parameters(), lr=0.001, amsgrad=True)

    uber_net.train()

    net.eval()

    _require_grad_list = []

    for _param in net.parameters():
        _require_grad_list.append((_param.requires_grad is True))
        _param.requires_grad = False

    input_data = torch.from_numpy(_im).float().to(network_components.nn_device)

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

        outputs = normalizing_layer(raw_output)

        entropy = torch.mean(torch.sum(-outputs * torch.log(outputs + 1.0e-16), dim=1))

        predicted = torch.round(outputs)

    for _param, _rg in zip(net.parameters(), _require_grad_list):
        _param.requires_grad = _rg

    return predicted.cpu().numpy(), entropy.cpu().numpy(), entropy_list


def crop_file_name(full_file_name):

    last_slash_ind = len(full_file_name) - 1

    while full_file_name[last_slash_ind] != "/":
        last_slash_ind -= 1

    cropped_name = full_file_name[last_slash_ind + 1:-4]

    return cropped_name


def predict_from_image_file(file_name, net, dataset, n_optim_iter=0,
                            brightness_scaling_method="whitened", verbose=True) -> dict:
    """
    Performs prediction using NanobarcodeNet *net*
    :param file_name: Filename for the image to be processed
    :param net: Trained network model used for prediction
    :type net: NanobarcodeClassifierNet
    :param dataset: Dataset handler
    :type dataset: dataset_handler.NanobarcodeDataset
    :param n_optim_iter: Number of entropy enhancing training done
    :param brightness_scaling_method: The method used for scaling brightness values
            (see *image_processing.scale_brightness*)
    :param verbose: If True, will print information
    :type verbose: bool

    :return: Dictionary of results
    """

    ind_filter = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]

    cropped_filename = crop_file_name(file_name)

    protein_colormap = image_processing.protein_colormap(dataset=dataset)
    cell_color = np.array([0.3, 0.3, 0.3, 1.0])

    protein_name = "UNKNOWN"

    for _pr in dataset.protein_names:
        if cropped_filename.find(_pr) != -1:
            protein_name = _pr

    if protein_name == "UNKNOWN" and cropped_filename.find("MOCK") != -1:
        protein_name = "Blank"

    if verbose:
        if protein_name != "UNKNOWN":
            print(f"Protein with name {protein_name} detected from filename!")
        else:
            print(f"Could not determine the protein name from filename.")

    raw_image_from_file = tifffile.imread(file_name)

    if raw_image_from_file.ndim == 3:
        raw_image_stack = raw_image_from_file.copy().reshape((1, *raw_image_from_file.shape))
    else:
        raw_image_stack = raw_image_from_file.copy()

    brightfield_stack = raw_image_stack[:, 4, :, :].copy().astype(np.float32)
    raw_image_stack = raw_image_stack[:, ind_filter, :, :].copy().astype(np.float32)

    segmentation_func = image_processing.KPCASeparate(raw_image_stack, threshold=0.95)
    # raw_image_brightness = np.clip(improc.scale_image(np.sum(np.abs(raw_image_stack), axis=1), "linear"), 0.0, 1.0)

    image_size = brightfield_stack.shape[1:]

    false_color_stack = []
    cell_halo_false_color_stack = []
    unprocessed_false_color_stack = []
    entropy_stack = []
    entropy_iter_stack = []

    # We consider the prediction precision in two steps:
    #
    #   1. The segmentation precision: are all the protein-containing pixels been correctly picked?
    #      No matter the type of the protein.
    #   2. The protein identification precision: is the protein in the image identified correctly?

    correct_protein_pick_with_segmentation = 0.0
    wrong_protein_pick_with_segmentation = 0.0

    correct_protein_pick = 0.0
    wrong_protein_pick = 0.0

    for image_slice, brightfield_slice in zip(raw_image_stack, brightfield_stack):

        if protein_name != "Blank":
            segmented_foreground_mask = segmentation_func(image_slice)
        else:
            segmented_foreground_mask = np.ones(image_slice.shape[-2:])

        segmented_background_mask = np.logical_not(segmented_foreground_mask)

        image_slice_scaled = image_processing.scale_brightness(image_slice, brightness_scaling_method)

        false_color_image = np.zeros((*image_size, 4))
        unprocessed_false_color_image = np.zeros((*image_size, 4))

        unprocessed_false_color_image[:, :, 3] = 1.0

        predicted, entropy, entropy_iter = feed_to_network_and_optimize_channel_scaling(net,
                                                                                        image_slice_scaled,
                                                                                        n_optim_iter)

        entropy_stack.append(entropy)
        entropy_iter_stack.append(entropy_iter.copy())

        for data_protein_name in dataset.protein_names:

            protein_id = dataset.brightness_data[data_protein_name]["ID"]

            predicted_foreground = predicted[:, protein_id].copy().reshape(image_size)
            # predicted_background = 1.0 - predicted_foreground

            for ch in range(4):
                false_color_image[:, :, ch] += predicted_foreground * protein_colormap[data_protein_name][ch]

            # the intended protein in the single-transfect slice
            if data_protein_name == protein_name:

                correct_protein_pick_with_segmentation += np.sum(predicted_foreground * segmented_foreground_mask)
                wrong_protein_pick_with_segmentation += np.sum(predicted_foreground * segmented_background_mask)

                correct_protein_pick += np.sum(predicted_foreground)

                # make a simple false-color image based on the
                # brightness of the original images when the protein-name is known
                for ch in range(3):
                    unprocessed_false_color_image[:, :, ch] =\
                        segmented_foreground_mask * protein_colormap[protein_name][ch]

            elif data_protein_name != 'Blank':

                wrong_protein_pick_with_segmentation += np.sum(predicted_foreground)
                wrong_protein_pick += np.sum(predicted_foreground)

        #             false_color_image = scale_image_linear(false_color_image)

        #         plt.imshow(brightfield_slice, cmap='bone')
        #         plt.axis('off')

        # Getting the cell halos from the bright-field image
        cell_halo = image_processing.scale_brightness(image_processing.get_cell_background(brightfield_slice), "linear")

        cell_halo_false_color_image = false_color_image.copy()

        for ch in range(4):
            cell_halo_false_color_image[:, :, ch] += cell_halo[:, :] * cell_color[ch]

        false_color_image = np.clip(false_color_image, 0.0, 1.0)
        cell_halo_false_color_image = np.clip(cell_halo_false_color_image, 0.0, 1.0)

        false_color_stack.append(false_color_image.copy())
        cell_halo_false_color_stack.append(cell_halo_false_color_image.copy())
        unprocessed_false_color_stack.append(unprocessed_false_color_image.copy())

    result = {"cropped filename": cropped_filename, "protein name": protein_name,
              "false-color stack": (np.array(false_color_stack) * 65535.0).astype(np.uint16),
              "cell-halo false-color stack": (np.array(cell_halo_false_color_stack) * 65535.0).astype(np.uint16),
              "unprocessed false-color stack": (np.array(unprocessed_false_color_stack) * 65535.0).astype(np.uint16),
              "entropy": np.array(entropy_stack), "iterative entropy": np.array(entropy_iter_stack),
              "precision (with segmentation)": correct_protein_pick_with_segmentation / max(
                      correct_protein_pick_with_segmentation + wrong_protein_pick_with_segmentation, 1.0E-16),
              "precision": correct_protein_pick / max(correct_protein_pick + wrong_protein_pick, 1.0E-16)}

    return result


def render_tomogram(false_image_color_stack, figure, pixel_size_z):

    assert false_image_color_stack.shape[1] == false_image_color_stack.shape[2]
    assert isinstance(figure, matplotlib.pyplot.Figure)

    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, false_image_color_stack.shape[1]),
                         np.linspace(0.0, 1.0, false_image_color_stack.shape[2]))

    ax = figure.add_subplot(projection='3d', frame_on=True)

    figure.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)

    z = 0.0

    # pixel size in x and y direction is 300 nm
    dz = 1.0 / float(false_image_color_stack.shape[1]) * pixel_size_z / 300.0  # * 0.5
    # dz = 0.03 * 17.0 / float(false_color_stack.shape[0])

    ax.set_facecolor('black')

    _box_color = 'white'

    _z = 0.0

    ax.plot3D([0.0, 0.0], [0.0, 1.0], [_z, _z], color=_box_color, zorder=-1)
    ax.plot3D([0.0, 1.0], [1.0, 1.0], [_z, _z], color=_box_color, zorder=-1)
    ax.plot3D([1.0, 1.0], [1.0, 0.0], [_z, _z], color=_box_color, zorder=-1)
    ax.plot3D([1.0, 0.0], [0.0, 0.0], [_z, _z], color=_box_color, zorder=-1)

    px = np.empty(0)
    py = np.empty(0)
    pz = np.empty(0)
    pc = np.empty((0, 4))

    last_z_order = 0

    for _zorder, _image in enumerate(false_image_color_stack[:, :, :, :]):

        _color = _image.copy().astype(np.float64) / 65535.0

        valid = np.sum(_color[:, :, :3], axis=2) > 0.1

        zz = np.ones_like(xx) * z

        # print(float(zz[valid].shape[0]) / float(np.prod(zz.shape)))

        ax.plot3D([0.0, 0.0], [0.0, 0.0], [z, z + dz], color=_box_color, zorder=_zorder)
        ax.plot3D([0.0, 0.0], [1.0, 1.0], [z, z + dz], color=_box_color, zorder=_zorder)
        ax.plot3D([1.0, 1.0], [1.0, 1.0], [z, z + dz], color=_box_color, zorder=_zorder)
        ax.plot3D([1.0, 1.0], [0.0, 0.0], [z, z + dz], color=_box_color, zorder=_zorder)

        z += dz

        px = np.concatenate((px, xx[valid]))
        py = np.concatenate((py, yy[valid]))
        pz = np.concatenate((pz, zz[valid]))
        pc = np.concatenate((pc, _color[valid]), axis=0)

        last_z_order = _zorder

    ax.scatter3D(px, py, pz,
                 c=pc,
                 edgecolor='none',
                 marker='.', s=4.0, alpha=0.5,
                 depthshade=True, zorder=last_z_order)

    _z = z

    # print(f"max z = {z * float(false_color_stack.shape[1])} pixels")

    ax.plot3D([0.0, 0.0], [0.0, 1.0], [_z, _z], color=_box_color, zorder=_zorder + 2)
    ax.plot3D([0.0, 1.0], [1.0, 1.0], [_z, _z], color=_box_color, zorder=_zorder + 2)
    ax.plot3D([1.0, 1.0], [1.0, 0.0], [_z, _z], color=_box_color, zorder=_zorder + 2)
    ax.plot3D([1.0, 0.0], [0.0, 0.0], [_z, _z], color=_box_color, zorder=_zorder + 2)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_zlim(-0.51, 0.51)
    ax.set_axis_off()

    # scale = np.diag([1.0, 1.0, 0.1, 1.0])

    # def short_proj ():
    #    return np.dot(Axes3D.get_proj(ax), scale)

    # ax.get_proj = short_proj
    ax.view_init(20, 40)
    ax.dist = 9.0

    ax.set_rasterized(True)
