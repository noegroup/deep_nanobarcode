import numpy as np
import scipy as sp
import skimage.morphology as skmorph
from skimage import restoration


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

