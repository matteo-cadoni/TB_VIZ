import numpy as np
from matplotlib.colors import Normalize
from utils.zarr_helper import create_rgb_zarr, create_tb_ome_zarr


def get_meta_data(reader, num_images):
    """ get and save meta-data of each image in the list that forms the bigger image.
    meta-data consists of x,y position of the image in the bigger image,
    and the image name and the line in which it is located in the bigger image.

    Args:
    reader: CziReader object
    num_images: int, number of images in the list that forms the bigger image

    """
    meta = np.zeros((num_images, 3))
    current_line = 0
    meta[0, 2] = 0
    # every tile has a position in the bigger image
    for i in range(0, num_images):

        x, y = reader.get_mosaic_tile_position(i)

        meta[i, 0] = x
        meta[i, 1] = y
        # tiles are stored from top to bottom, left to right
        meta[i, 2] = current_line

        if i > 0:
            # if there is a big jump in x position, then the tile is in a new line
            if np.abs(meta[i, 0] - meta[i - 1, 0]) > 1000:
                current_line += 1
                meta[i, 2] = current_line
    return meta


def reproduce_map(meta):
    """ Reproduce a metadata map from the meta data of the images.
    Metadata can be used as a heatmap to show bacilli count on top of the real image.
    This is based on the assumption that tiles in the middle will fill the entire image from left to right,
    forming a maximum_line (full line) in the middle of the image (or multiple maximum lines).

    Args:
        meta: numpy array, metadata of the images

    Returns:
        numpy array, reproduced map

    """

    # get the maximum number of tiles in a line
    max_tiles = max([sum(meta[:, 2] == i) for i in range(0, int(np.max(meta[:, 2]).item() + 1))]).item()
    num_lines = int(np.max(meta[:, 2]))
    reproduced_map = np.zeros((num_lines + 1, max_tiles))
    # get all the lines that have the maximum number of tiles
    maximum_lines = [i for i in range(0, num_lines + 1) if sum(meta[:, 2] == i) == max_tiles]

    # starting from middle go up to fill the map
    offset_of_previous_line = 0
    for i in range(maximum_lines[0], 0, -1):
        tiles = np.where(meta[:, 2] == i - 1)[0]
        old_tiles = np.where(meta[:, 2] == i)[0]
        num_tiles = tiles.size
        num_old_tiles = old_tiles.size
        # go over the previous line and find the tiles that are close to the current line
        for h in range(0, num_old_tiles):
            # neighbouring tiles have a small difference in y position
            if np.abs(meta[tiles[0], 1] - meta[old_tiles[0] + h, 1]) < 100:
                reproduced_map[i - 1, offset_of_previous_line + h: num_tiles + offset_of_previous_line + h] = tiles + 1
                offset_of_previous_line += h
                break

    # starting from middle go down to fill the map
    offset_of_previous_line = 0
    for i in range(maximum_lines[-1], num_lines):
        tiles = np.where(meta[:, 2] == i + 1)[0]
        old_tiles = np.where(meta[:, 2] == i)[0]
        num_tiles = tiles.size
        num_old_tiles = old_tiles.size
        # go over the previous line and find the tiles that are close to the current line
        for h in range(0, num_old_tiles):
            # neighbouring tiles have a small difference in y position
            if np.abs(meta[tiles[0], 1] - meta[old_tiles[0] + h, 1]) < 100:
                reproduced_map[i + 1, offset_of_previous_line + h: num_tiles + offset_of_previous_line + h] = tiles + 1
                offset_of_previous_line += h
                break

    # assign also the lines that have exactly the same number of tiles as the maximum number of tiles
    for i in range(0, num_lines):
        if sum(meta[:, 2] == i) == max_tiles:
            reproduced_map[i, :] = np.where(meta[:, 2] == i)[0] + 1

    return reproduced_map


def reproduce_map_of_original_size(original_image_shape, filled_map):
    """ Reproduce the filled map to the original image size.
    Args:
        original_image_shape: tuple, shape of the original image
        filled_map: numpy array, filled map of the original image, filled with bacili count x tile

    Returns:
        numpy array, filled map of the original image, filled with bacilli count x pixel
    """
    original_image_size = original_image_shape
    tile_size = (original_image_shape[0] // filled_map.shape[0], original_image_shape[1] // filled_map.shape[1])
    filled_map_og_size = np.zeros(original_image_size)
    # make every entry of the filled map a tile of shape tile_size filled with the entry value
    for i in range(0, filled_map.shape[0]):
        for j in range(0, filled_map.shape[1]):
            filled_map_og_size[i * tile_size[0]: (i + 1) * tile_size[0],
            j * tile_size[1]: (j + 1) * tile_size[1]] = filled_map[i, j]

    return filled_map_og_size


def get_map(reader, bacilli_per_tile):
    """ Get the original image and the heatmap of the bacilli count on the original image.

    Args:
        reader: CziReader object
        bacilli_per_tile: dictionary, bacilli count per tile

    Returns:
        tuple of two strings, name of the zarr files containing the original image and the heatmap
    """

    # get images as list and the entire image
    image_list = reader.get_image_dask_data("MYX", S=3)
    data = reader.mosaic_data

    image_shape = data.shape
    image_shape = image_shape[1:]
    data = data.reshape((1,) + data.shape)

    # normalize data to 0-255 (uint8) in chunks for memory efficiency
    data_min = data.min()
    data_max = data.max()
    chunk_size_x = 1024
    chunk_size_y = 1024
    normalized_data = np.empty(data.shape, dtype=np.uint8)

    for y_start in range(0, data.shape[2], chunk_size_y):
        for x_start in range(0, data.shape[3], chunk_size_x):
            y_end = min(y_start + chunk_size_y, data.shape[2])
            x_end = min(x_start + chunk_size_x, data.shape[3])

            chunk = data[0, 0, y_start:y_end, x_start:x_end].astype(np.float32)
            chunk_normalized = (chunk - data_min) / (data_max - data_min)
            chunk_unit8 = (chunk_normalized * 255).astype(np.uint8)
            normalized_data[0, 0, y_start:y_end, x_start:x_end] = chunk_unit8
    del data

    # create grayscale zarr of numpy array of the original image
    normalized_data = np.squeeze(normalized_data, axis=(0, 1))
    create_tb_ome_zarr(normalized_data, "original_image.zarr")
    del normalized_data

    # get heatmap using the bacilli information
    meta = get_meta_data(reader, len(image_list))
    num_images = len(image_list)
    del image_list
    reproduced_map = reproduce_map(meta)
    bacilli = [bacilli_per_tile[i] for i in range(num_images)]
    # fill the map with the bacilli count
    filled_map = np.zeros(reproduced_map.shape)
    for i in range(len(bacilli)):
        filled_map[reproduced_map == i + 1] = bacilli[i]
    # get the heatmap of the bacilli count in the original image size
    filled_map_og_size = reproduce_map_of_original_size(image_shape, filled_map)
    norm = Normalize(vmin=filled_map_og_size.min(), vmax=filled_map_og_size.max())

    # make the heatmap in chunks for memory efficiency
    gray_heatmap = np.zeros((filled_map_og_size.shape[0], filled_map_og_size.shape[1]), dtype=np.uint8)
    chunk_sizes = 1096
    for i in range(0, filled_map_og_size.shape[0], chunk_sizes):
        for j in range(0, filled_map_og_size.shape[1], chunk_sizes):
            chunk_float32 = filled_map_og_size[i: min(i + chunk_sizes, filled_map_og_size.shape[0]),
                            j:min(j + chunk_sizes, filled_map_og_size.shape[1])]
            normalized_chunk = norm(chunk_float32)
            gray_heatmap[i:min(i + chunk_sizes, filled_map_og_size.shape[0]),
            j:min(j + chunk_sizes, filled_map_og_size.shape[1])] = (normalized_chunk * 255).astype(np.uint8)

    # create zarr directory for grayscale heatmap
    create_tb_ome_zarr(gray_heatmap, "heatmap_image.zarr")

    return "original_image.zarr", "heatmap_image.zarr"
