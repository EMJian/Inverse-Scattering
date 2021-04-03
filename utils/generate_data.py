import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from scatterers.mnist import MNISTScatterer
from forward_models.mom import MethodOfMomentModel


class GenerateData:

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    scatterer_path = os.path.join(data_path, "scatterer_data")
    field_path = os.path.join(data_path, "field_data")

    @staticmethod
    def generate_mnist_scatterers(digit, how_many=None):
        scatterers = []
        digits = mnist.load_data()
        training_data = digits[0]
        train_X = training_data[0]
        train_Y = training_data[1]
        images = [x for i, x in enumerate(train_X) if train_Y[i] == digit]

        if how_many:
            images = images[:how_many]
        for i in range(len(images)):
            image = images[i]
            plt.imshow(image)
            scatterer = MNISTScatterer(image)
            scatterers.append(scatterer.get_scatterer_profile())

        filename = f"scatterers_mnist_{digit}_{how_many}"
        np.savez(os.path.join(GenerateData.scatterer_path, filename), scatterers=scatterers)
        return scatterers

    @staticmethod
    def generate_circular_scatterers():
        pass

    @staticmethod
    def generate_forward_data(scatterers):
        incident_fields = []
        incident_powers = []
        total_fields = []
        total_powers = []
        for i, scatterer in enumerate(scatterers):
            model = MethodOfMomentModel()
            incident_power, total_power, direct_field, total_field = \
                model.generate_forward_data(scatterer)
            incident_fields.append(direct_field)
            incident_powers.append(incident_power)
            total_fields.append(total_field)
            total_powers.append(total_power)
        return incident_fields, incident_powers, total_fields, total_powers

    @staticmethod
    def save_forward_data(incident_fields, incident_powers, total_fields, total_powers, scatterer_type, spec=""):
        filename = f"forward_data_{scatterer_type}_{spec}"
        np.savez(os.path.join(GenerateData.field_path, filename),
                 incident_fields=incident_fields,
                 incident_powers=incident_powers,
                 total_fields=total_fields,
                 total_powers=total_powers,
                 )
        print("Data saved")


if __name__ == '__main__':

    scatterer_type = "mnist"
    digit = 5
    how_many = 2
    scatterers = GenerateData.generate_mnist_scatterers(digit, how_many)
    incident_fields, incident_powers, total_fields, total_powers = GenerateData.generate_forward_data(scatterers)
    GenerateData.save_forward_data(incident_fields, incident_powers, total_fields, total_powers,
                                   scatterer_type, spec=f"{digit}_{how_many}")
