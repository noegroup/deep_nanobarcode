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


import numpy as np
import scipy as sp
import skimage.morphology as skmorph
from skimage import restoration
from sklearn.decomposition import KernelPCA

def scale_image(image, scaling="fixed"):

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
        raise ValueError("Type of dataset scaling is unknown!")


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

    brightfield_image_enhanced = scale_image(
        np.exp(-scale_image(np.real(np.fft.ifft2(bff)), "linear") ** 2 / 0.2), "linear")

    seed = np.copy(brightfield_image_enhanced)
    seed[1:-1, 1:-1] = brightfield_image_enhanced.min()
    mask = brightfield_image_enhanced

    dilated = skmorph.reconstruction(seed, mask, method='dilation')
    cells = np.clip((scale_image(brightfield_image_enhanced - dilated, "linear") - 0.03) * 200.0, 0.0, 1.0)

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

