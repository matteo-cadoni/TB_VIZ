
import cv2 as cv


class Postprocessing:
    """Class that cleans imprecision after thresholding.
    Different thresholding algorithms have different cleaning methods.
    Split otsu thresholding is the only one that needs to split the image into tiles.
    The adaptive methods only need morphological operations.

     attributes
     ----------
    img:
        image to be cleaned
    config:
        dictionary with the parameters
    tiles:
        list with the different small tiles

    methods
    -------
    split_into_tiles(tile_size=16):
        split image into tiles of shape tile_size * tile_size
    cleaning_tiles():
        Clean the small tiles of the image
    check_image(img: np.ndarray):
        For every sub-image we check if there is a bacilli or not
    reconstruct_image():
        Reconstruct the image from the clean sub-tiles
    remove_noise():
        Remove noise from the image
    apply():
        Apply the postprocessing to the image
    """

    def __init__(self, img):
        """
        parameters
        ----------
        img:
            image to be cleaned
        """
        self.img = img

    def remove_noise(self):
        """ Perform morphological opening and closing to remove noise

        returns
        -------
        closing:
            cleaned imag
        """

        # define kernel for opening
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        # perform morphological opening to remove background noise
        opening = cv.morphologyEx(self.img, cv.MORPH_OPEN, kernel)

        # perform morphological closing to close small holes inside foreground objects
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        return closing

    def apply(self):
        """ Apply the postprocessing to the image

        returns
        -------
        whole_img_not_cleaned:
            image before cleaning
        whole_img_cleaned:
            image after cleaning
        num_bacilli:
            number of bacilli in the image
        """

        whole_img_cleaned = self.remove_noise()
        num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(whole_img_cleaned, connectivity=8)
        return self.img, whole_img_cleaned, num_labels - 1, stats
