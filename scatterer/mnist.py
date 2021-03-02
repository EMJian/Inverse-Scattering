import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize

from tensorflow.keras.datasets import mnist


class MNISTScatterer:

    def __init__(self, image, number_of_grids, object_permittivity):

        self.image = image
        self.object_permittivity = object_permittivity
        self.number_of_grids = number_of_grids

    def convert_to_binary(self):
        grayscale = rgb2gray(self.image)
        threshold = threshold_otsu(grayscale)
        self.binary = grayscale > threshold

    def resize_image(self):
        self.resized = resize(self.binary, (self.number_of_grids, self.number_of_grids))

    def set_grid_permittivities(self):
        self.scatterer = np.ones((self.number_of_grids, self.number_of_grids), dtype=float)
        self.scatterer[self.resized > 0] = self.object_permittivity

    def view_images(self):
        plt.imshow(self.image, cmap=plt.cm.brg)
        plt.imshow(self.binary, cmap=plt.cm.gray)
        plt.imshow(self.resized, cmap=plt.cm.gray)
        plt.imshow(self.scatterer, cmap=plt.cm.gray)

    def get_scatterer_profile(self):
        self.convert_to_binary()
        self.resize_image()
        self.set_grid_permittivities()
        self.view_images()
        return self.scatterer


if __name__ == '__main__':

    digits = mnist.load_data()
    digit_images = digits[0][0]
    permittivity = []

    for i in range(10):
        image = digit_images[0, :, :]
        scatterer = MNISTScatterer(image, 100, 3)
        permittivity.append(scatterer.get_scatterer_profile())

    permittivity = np.reshape(permittivity, (10, 100, 100))
    print(permittivity.shape)
