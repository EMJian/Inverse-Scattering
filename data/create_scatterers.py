import os
import numpy as np
from tensorflow.keras.datasets import mnist

from scatterers.mnist import MNISTScatterer

if __name__ == '__main__':

    digit = 8

    permittivity = []

    digits = mnist.load_data()
    training_data = digits[0]
    train_X = training_data[0]
    train_Y = training_data[1]
    images_three = [x for i, x in enumerate(train_X) if train_Y[i] == digit]

    for i in range(len(images_three[:10])):
        image = images_three[i]
        scatterer = MNISTScatterer(image, 100, 3)
        permittivity.append(scatterer.get_scatterer_profile())

    script_dir = os.path.dirname(__file__)
    filename = f"mnist_scatterers_{digit}"
    np.savez(os.path.join(script_dir, "scatterer_data", filename), mnist_scatterers_8=permittivity)
