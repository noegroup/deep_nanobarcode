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
import numpy
import numpy as np
import scipy as sp
import warnings
import skimage.morphology as skmorph
from skimage import restoration
from sklearn.decomposition import KernelPCA
from matplotlib import colors as mpl_colors
from matplotlib import cm as mpl_cm


def scale_brightness(image, scaling="fixed") -> numpy.ndarray:
    """

    :param image:
    :param scaling: type of scaling applied to the brightness
            can be any of:

            - "fixed": Divides the brightness values by 65535 (assumes 16-bit channels)
            - "linear": Linearly scales brightness values between 0 and 1
            - "whitened": Removes the mean and divides by standard deviation of brightness
            - "channel_whitened": Similar to "whitened", but applied to each channel separately
    :type scaling: str
    :return: copy of the image with brightness values scaled
    """

    if scaling == "fixed":
        return image / 65535.0
    elif scaling == "linear":
        _max = np.amax(image)
        _min = np.amin(image)

        if _max > _min:
            return (image - _min) / (_max - _min)
        else:
            return image - _min
    elif scaling == "whitened":
        return (image - np.mean(image)) / np.std(image)
    elif scaling == "channel_whitened":
        _im = image.copy()

        channel_mean = np.mean(image, axis=1)
        channel_std = np.std(image, axis=1)

        assert _im.shape[0] == 10

        for i in range(_im.shape[0]):
            _im[i, :] = (_im[i, :] - channel_mean[i]) / channel_std[i]

        return _im
    else:
        raise ValueError("Type of brightness scaling is unknown!")


def mpl_color(color_name) -> list:
    """
    Returns the RGB components of a named color from matplotlib colors.

    :param color_name: Color name accepted by matplotlib colors
    :return: list of RGBA values
    """

    hex_color = mpl_colors.get_named_colors_mapping()[color_name]
    h = hex_color.lstrip('#')

    rgba = list(float(int(h[i:i + 2], 16)) / 255.0 for i in (0, 2, 4))
    rgba.append(1.0)

    return rgba


def protein_colormap(dataset=None, use_generic_colormap=False) -> dict:
    """
    Assigns a map of false colors to a set of proteins.
    If a data_handler is provided, it will be checked if
    all the proteins in the data_handler

    :param dataset: Dataset handlers used in training the network
    :type dataset: dataset_handler.NanobarcodeDataset
    :param use_generic_colormap: If Ture, a sequential colormap is used to assign colors
    :type use_generic_colormap: bool
    :return: protein_color
    """

    protein_color = dict()

    if use_generic_colormap:

        _colormap = mpl_cm.tab20(np.linspace(0.0, 1.0, dataset.n_proteins))

        for protein_name, __color in zip(dataset.protein_names, _colormap):
            protein_color[protein_name] = __color

    else:

        protein_color["Vti1a"] = mpl_color('xkcd:magenta')
        protein_color["GFP"] = mpl_color('xkcd:lime')
        protein_color["TOM70"] = mpl_color('xkcd:lilac')  # mpl_color('xkcd:slate blue')
        protein_color["SNAP25"] = mpl_color('xkcd:burgundy')
        protein_color["Rab5a"] = mpl_color('xkcd:light orange')
        protein_color["STX4"] = mpl_color('xkcd:dark orange')
        protein_color["STX6"] = mpl_color('xkcd:light red')
        protein_color["LifeAct"] = mpl_color('xkcd:dark sky blue')  # mpl_color('xkcd:navy')
        protein_color["NLS"] = mpl_color('xkcd:dark yellow')
        protein_color["KDEL"] = mpl_color('xkcd:moss')
        protein_color["GalNact"] = mpl_color('xkcd:beige')
        protein_color["ENDO"] = mpl_color('xkcd:bright yellow')  # mpl_color('xkcd:blue')
        protein_color["Blank"] = np.array([0.0, 0.0, 0.0, 1.0])

    for protein_name in dataset.protein_names:
        if list(protein_color.keys()).index(protein_name) == -1:
            warnings.warn(f"No color has been assigned to the protein {protein_name} from the dataset!")

    return protein_color


class KPCASeparate(object):

    def __init__(self, image, threshold=0.75):

        self.model = KernelPCA(n_components=2,
                               kernel='cosine', gamma=None, degree=3, coef0=1,
                               kernel_params=None, alpha=1.0,
                               fit_inverse_transform=False,
                               eigen_solver='auto', tol=0, max_iter=None,
                               iterated_power='auto', remove_zero_eig=False,
                               random_state=None, copy_X=True, n_jobs=None)

        self.threshold = threshold

        if image.ndim not in [3, 4]:
            raise ValueError("Image dimensions should be either 3: (channel, X, Y) or 4: (z-stack, channel, X, Y)!")

        if image.ndim == 3:
            image = image.reshape((1, *image.shape))

        n_channels = image.shape[1]

        image_wt = (image.copy() - np.mean(image)) / np.std(image)
        image_wt = np.transpose(image_wt, [1, 0, 2, 3])

        all_pixels = image_wt.reshape((n_channels, -1))

        n_pixels = all_pixels.shape[1]

        x = all_pixels[:, np.random.randint(0, n_pixels, 6000)].T.copy()
        y = self.model.fit_transform(x)
        self.y_min, self.y_max = np.amin(y[:, 0]), np.amax(y[:, 0])

    def __call__(self, image):

        if image.ndim != 3:
            raise ValueError("Image dimensions should be 3: (channel, X, Y)!")

        n_channels = image.shape[0]
        image_dim = image.shape[1:3]

        image_wt = (image.copy() - np.mean(image)) / np.std(image)

        segmented_pixels = self.model.transform(image_wt.reshape((n_channels, -1)).T)

        binary_mask = ((segmented_pixels[:, 0] - self.y_min) > self.threshold *
                       (self.y_max - self.y_min)).reshape(image_dim)

        return binary_mask


def get_cell_background(brightfield_image):

    brightfield_image = (brightfield_image - np.mean(brightfield_image)) / np.std(brightfield_image)

    bff = np.fft.fft2(brightfield_image)

    window = sp.signal.windows.gaussian(brightfield_image.shape[0], std=170.0)
    window = np.outer(window, window)

    bff *= window

    brightfield_image_enhanced = scale_brightness(
        np.exp(-scale_brightness(np.real(np.fft.ifft2(bff)), "linear") ** 2 / 0.2), "linear")

    seed = np.copy(brightfield_image_enhanced)
    seed[1:-1, 1:-1] = brightfield_image_enhanced.min(initial=0.0)
    mask = brightfield_image_enhanced

    dilated = skmorph.reconstruction(seed, mask, method='dilation')
    cells = np.clip((scale_brightness(brightfield_image_enhanced - dilated, "linear") - 0.03) * 200.0, 0.0, 1.0)

    cells = restoration.denoise_tv_bregman(cells, weight=0.4)

    #             binary_mask = (cells > filters.threshold_otsu(cells)).astype(np.float32)

    #             binary_mask = skmorph.erosion(binary_mask, selem2)
    #             binary_mask = skmorph.closing(binary_mask, selem1)
    # binary_mask = skmorph.dilation(binary_mask, selem2)

    #             markers = np.zeros(brightfield_image.shape, dtype=np.uint)
    #             markers[brightfield_image < 0.01] = 1
    #             markers[brightfield_image > 0.99] = 2

    #             cell_edge = random_walker(brightfield_image, markers, beta=10, mode='bf')

    #             cell_edge = scale_image(filters.roberts(brightfield_image_enhanced).astype(np.float32))
    #             cell_edge = cells#feature.canny(binary_mask).astype(np.float32)

    return cells

