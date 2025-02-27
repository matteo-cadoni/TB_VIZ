import zarr
import numpy as np


def create_rgb_zarr(rgb_ndarray, zarr_directory):
    """
    Create a zarr file from a 3D numpy array (RGB image)
    Parameters
    ----------
    rgb_ndarray : numpy.ndarray (3D)
        RGB image
    zarr_directory : str
        Path to save the zarr file

    Returns
    -------
    str
        Path to the zarr file

    """
    pyramid = create_pyramids_rgb(rgb_ndarray, 2, 6)
    new_pyramid = []
    # convert each level of pyramid into 5D image (t, c, z, y, x)
    for pixels in pyramid:
        # make last dimension the first
        pixels = np.moveaxis(pixels, -1, 0)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = np.expand_dims(pixels, axis=2)
        new_pyramid.append(pixels)

    store = zarr.DirectoryStore(zarr_directory)
    grp = zarr.group(store, overwrite=True)
    paths = []
    for path, dataset in enumerate(new_pyramid):
        grp.create_dataset(str(path), data=new_pyramid[path])
        paths.append({"path": str(path)})

    image_data = {
        "id": 1,
        "channels": [
            {
                "color": "FF0000",
                "window": {"start": 0, "end": 255},
                "label": "Red",
                "active": True,
            },
            {
                "color": "00FF00",
                "window": {"start": 0, "end": 255},
                "label": "Green",
                "active": True,
            },
            {
                "color": "0000FF",
                "window": {"start": 0, "end": 255},
                "label": "Blue",
                "active": True,
            },
        ],
        "rdefs": {
            "model": "color",
        },
    }

    multiscales = [
        {
            "version": "0.1",
            "datasets": paths,
        }
    ]
    grp.attrs["multiscales"] = multiscales
    grp.attrs["omero"] = image_data

    return zarr_directory


def create_pyramid(image, downsample, max_layer):
    """ Create a pyramid of images, each downsampled by a factor of downsample from the previous one

    Parameters
    ----------
    image : numpy.ndarray
        Image to create pyramid from
    downsample : int
        Factor to downsample image by
    max_layer : int
        Number of layers in the pyramid

    Returns
    -------
    List
        List of images in the pyramid

    """
    pyramids = [image]
    for i in range(1, max_layer + 1):
        pyramids.append(pyramids[i - 1][::downsample, ::downsample])
    return pyramids


def create_pyramids_rgb(image, downsample, max_layer):
    """ Create a pyramid of images, each downsampled by a factor of downsample from the previous one
    but for RGB images

    Parameters
    ----------
    image : numpy.ndarray
        Image to create pyramid from
    downsample : int
        Factor to downsample image by
    max_layer : int
        Number of layers in the pyramid

    Returns
    -------
    List
        List of images in the pyramid

    """

    pyramids = [image]
    for i in range(1, max_layer + 1):
        pyramids.append(pyramids[i - 1][::downsample, ::downsample, :])
    return pyramids


def create_tb_ome_zarr(image, zarr_directory):
    """ Create a zarr file from a 2D numpy array (Grayscale image)

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image
    zarr_directory : str
        Path to save the zarr file

    """

    gaussian = create_pyramid(image, 2, 6)

    pyramid = []

    for pixels in gaussian:

        pixels = np.expand_dims(pixels, axis=0)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = np.expand_dims(pixels, axis=0)
        pyramid.append(pixels)

    store = zarr.DirectoryStore(zarr_directory)
    grp = zarr.group(store, overwrite=True)
    paths = []
    for path, dataset in enumerate(pyramid):
        grp.create_dataset(str(path), data=pyramid[path])
        paths.append({"path": str(path)})

    image_data = {
        "id": 1,
        "channels": [
            {
                "color": "FFFFFF",  # Grayscale (white for visualization)
                "window": {"start": 0, "end": 255},
                "label": "Gray",  # Label for the grayscale channel
                "active": True,  # Keep active if required
            },
        ],
        "rdefs": {
            "model": "greyscale",
        },
    }

    multiscales = [
        {
            "version": "0.1",
            "datasets": paths,
        }
    ]
    grp.attrs["multiscales"] = multiscales
    grp.attrs["omero"] = image_data
