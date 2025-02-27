import cv2 as cv


class Thresholding:
    """
    perform thresholding on each tile
    """

    def __init__(self, img):
        self.img = img

    def adpt_g_thresholding(self, block_size: int, c: int):
        """
        Threshold and binarize an image using adaptive thresholding using a Gaussian weighted sum

        :param block_size: parameter for adaptive thresholding
        :param c: parameter for adaptive thresholding
        :return: th: binary image
        """
        # The threshold value is a gaussian-weighted sum of the neighbourhood (here of size 25)
        # values minus the constant C (which is set to -7)
        thresholded_image = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                 block_size, c)
        return thresholded_image

    def apply(self):
        return self.adpt_g_thresholding(25, -7)
