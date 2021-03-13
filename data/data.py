import os
import numpy as np
from tensorflow.keras.datasets import mnist

from scatterer.mnist import MNISTScatterer

if __name__ == '__main__':

    permittivity = []

    digits = mnist.load_data()
    training_data = digits[0]
    train_X = training_data[0]
    train_Y = training_data[1]
    images_three = [x for i, x in enumerate(train_X) if train_Y[i] == 3]

    for i in range(len(images_three)):
        image = images_three[i]
        scatterer = MNISTScatterer(image, 100, 3)
        permittivity.append(scatterer.get_scatterer_profile())

    script_dir = os.path.dirname(__file__)
    filename = "mnist_scatterers_3"
    np.savez(os.path.join(script_dir, filename), mnist_scatterers_3=permittivity)
