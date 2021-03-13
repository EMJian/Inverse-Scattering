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
        fig = plt.figure(figsize=(8, 8))

        image = fig.add_subplot(2, 2, 1)
        plt.imshow(self.image, cmap=plt.cm.brg)
        image.title.set_text("Original RGB image: 28x28 pixels")

        binary = fig.add_subplot(2, 2, 2)
        plt.imshow(self.binary, cmap=plt.cm.gray)
        binary.title.set_text("Binary image: 28x28 pixels")

        resized = fig.add_subplot(2, 2, 3)
        plt.imshow(self.resized, cmap=plt.cm.gray)
        resized.title.set_text("Resized binary image: 100x100 pixels")

        final = fig.add_subplot(2, 2, 4)
        plt.imshow(self.scatterer, cmap=plt.cm.gray, extent=[0, 0.5, 0, 0.5])
        final.title.set_text("Resized scatterer: 100x100 pixels")

        plt.show()

    def get_scatterer_profile(self):
        self.convert_to_binary()
        self.resize_image()
        self.set_grid_permittivities()
        return self.scatterer


if __name__ == '__main__':

    digits = mnist.load_data()
    digit_images = digits[0][0]
    permittivity = []

    for i in range(1):
        image = digit_images[0, :, :]
        scatterer = MNISTScatterer(image, 100, 3)
        permittivity.append(scatterer.get_scatterer_profile())
        scatterer.view_images()
