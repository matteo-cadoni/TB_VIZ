
import os
from aicsimageio.readers import CziReader
import h5py


def load(czi_path: str):
    """Load the data from the config file.
    Initialization function for the Loader class.

    parameters:
    ----------
    load_config: config file
    """
    loader = Loader(czi_path)
    loader.load()
    img = loader.data_array

    return img, loader


class Loader:
    """Class that given a single sputum smear image made up of multiple tiles,
     loads and transform it to numpy array.

    attributes
    ----------
    czi_path
        path to the czi file
    tile
        tile to be loaded or None
    dataset_name
        name of the dataset when saved
    data_array
        numpy array containing the image

    methods
    -------
    read_array_from_h5(h5_path)
        read the array from h5 file
    save_array_to_h5(h5_path)
        save the array to h5 file
    read_array_from_czi()
        read the array from czi file
    load()
        load the array from h5 file if it exists, otherwise read from czi file
    """

    def __init__(self, czi_path):
        """
        parameters
        ----------
        czi_path
            path to the czi file
        tile
            number of tile to be loaded or None
        """
        self.czi_path = czi_path

        self.dataset_name = f'smear_{czi_path}'

    def read_array_from_h5(self, h5_path):
        """read the array from h5 file

        parameters
        ----------
        h5_path
            path to the h5 file
        """
        h5file = h5py.File(h5_path, 'r')
        self.data_array = h5file[self.dataset_name][:]

    def save_array_to_h5(self, h5_path):
        """" save the array to h5 file

        parameters
        ----------
        h5_path:
            path to the h5 file
        """

        h5file = h5py.File(h5_path, 'w')
        h5file.create_dataset(self.dataset_name, data=self.data_array)

    def read_array_from_czi(self):
        """ read the array from czi file

        """

        reader = CziReader(self.czi_path)
        self.reader = reader
        self.data_array = reader.get_image_data("MYX", C=0)

    def load(self):
        """ load the array from h5 file if it exists, otherwise read from czi file

        """

        # check if h5_file exists, otherwise create it
        h5_path = os.path.join('h5_data', self.dataset_name + '.h5')
        if os.path.isfile(h5_path):

            self.read_array_from_h5(h5_path)

        else:

            if not os.path.exists('h5_data'):

                os.makedirs('h5_data')
            assert os.path.isfile(self.czi_path), "Please specify the path to the czi file"
            self.read_array_from_czi()
