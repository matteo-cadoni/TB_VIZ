import numpy as np


def rescale(img):
    """rescale and convert image to uint8

    returns
    -------
    rescaled_image : numpy.ndarray
        rescaled image
    """
    rescaled_image = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    rescaled_image = np.round(rescaled_image).astype('uint8')
    return rescaled_image


def clean_stats(stats):
    """Delete connected components that are too small, and
    connected components that are too large.

    parameters:
    ----------
    stats: stats from connected components

    returns:
    -------
    stats1: cleaned stats
    """
    # make a copy of stats
    stats1 = stats.copy()
    # indices to delete
    indices = []
    # delete
    for i in range(0, stats.shape[0]):
        if stats[i, 4] > 625:
            # append index
            indices.append(i)

        if stats[i, 4] < 20:
            indices.append(i)
    stats1 = np.delete(stats1, indices, axis=0)
    return stats1
