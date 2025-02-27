""" Cropping functions, to crop 50x50 images from an image, where the bacillus is at the center of the image."""

import cv2 as cv
import numpy as np
from utils.generic_utils import clean_stats


def pad_images(image):
    """ This function will pad the images to be 50x50

    parameters
    ----------
    image: numpy array
        image to be padded

    returns
    -------
    padded_image: numpy array
        padded image
    """
    padded_image = np.zeros((50, 50))
    if image.shape[0] < 50 or image.shape[1] < 50:
        padded_image[0:image.shape[0], 0:image.shape[1]] = image

        return padded_image
    else:

        return image


class Cropping:
    """
    A class that can be used to crop 50x50 images from an image,
    where at the center of each image is a connected component,
    i.e. a suspected bacillus.

    attributes
    ----------
    original_tile: numpy array
        original tile image
    thresholded_img: numpy array
        thresholded tile image

    methods
    -------
    find_center_of_mass()
        Given the statistics of the connected components,
        find the center of mass of each component.
    crop_images(center_of_mass)
        Given the center of mass of each connected component,
        crop the original image and return a list of cropped images.
    crop_and_pad()
        Create a numpy array of cropped and padded images,
        given the center of mass of each connected component,
        pad the images if necessary.
    """

    def __init__(self, original_tile, thresholded_img):
        """ Initialize stats given a thresholded image.

        parameters
        ----------
        original_tile: numpy array
            original tile image
        thresholded_img: numpy array
            thresholded tile image
        """
        self.original_tile = original_tile
        # initialize stats
        num_labels, labels_im, self.stats, centroids = cv.connectedComponentsWithStats(
            np.uint8(thresholded_img), connectivity=8)

    def find_center_of_mass(self):
        """Given the statistics of the connected components,
        find the center of mass of each component.

        returns
        -------
        center_of_mass: list
            list of center of mass coordinates
        """
        center_of_mass = []
        # clean stats before finding center of mass
        self.stats = clean_stats(self.stats)
        # find all center of mass
        for i in range(1, self.stats.shape[0]):
            # get coordinates height and width of each component
            x = self.stats[i, 0]
            y = self.stats[i, 1]
            w = self.stats[i, 2]
            h = self.stats[i, 3]
            # append
            center_of_mass.append((x + w // 2, y + h // 2))
        return center_of_mass

    def crop_images(self, center_of_mass):
        """Given the center of mass of each connected component,
        crop the original image and return a list of cropped images

        parameters
        ----------
        center_of_mass: list
            list of center of mass coordinates

        returns
        -------
        cropped_images: list
            list of cropped images
        """
        cropped_images = []
        for i in range(len(center_of_mass)):
            x = center_of_mass[i][0]
            y = center_of_mass[i][1]
            cropped_img = self.original_tile[y-25:y+25, x-25:x+25]
            cropped_images.append(cropped_img)

        return cropped_images

    def crop_and_pad(self):
        """Create a numpy array of cropped and padded images,
        given the center of mass of each connected component

        returns
        -------
        cropped_numpy: numpy array
             array of cropped and padded images
        h : str
            "no images" if no images are found
        """

        center_of_mass = self.find_center_of_mass()
        cropped_images = self.crop_images(center_of_mass)

        # initialize numpy array
        if len(cropped_images) > 1:

            a = np.array(cropped_images[0])
            a = pad_images(a)
            b = np.array(cropped_images[1])
            b = pad_images(b)
            # stack first two images
            cropped_numpy = np.stack((a, b), axis=0)
            # concatenate the rest of the images
            for i, img in enumerate(cropped_images):
                if i > 1:
                    c = np.array(cropped_images[i])
                    c = pad_images(c)
                    cropped_numpy = np.concatenate((cropped_numpy, [c]), axis=0)
            return cropped_numpy
        elif len(cropped_images) == 1:
            a = np.array(cropped_images[0])
            a = pad_images(a)
            cropped_numpy = np.expand_dims(a, axis=0)
            return cropped_numpy
        else:
            h = "no images"
            return h
