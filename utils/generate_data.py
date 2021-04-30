import os
import numpy as np
from joblib import Parallel, delayed

from tensorflow.keras.datasets import mnist

from config import Config
from scatterers.mnist import MNISTScatterer
from forward_models.mom import MethodOfMomentModel
from inverse_models.linear.inverse import LinearInverse


class GenerateData:

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    scatterer_path = os.path.join(data_path, "scatterer_data")
    field_path = os.path.join(data_path, "field_data")
    initial_guess_path = os.path.join(data_path, "initial_guess_data")

    @staticmethod
    def generate_mnist_scatterers(digit, how_many=None, spec=""):
        scatterers = []
        digits = mnist.load_data()
        training_data = digits[0]
        train_X = training_data[0]
        train_Y = training_data[1]
        images = [x for i, x in enumerate(train_X) if train_Y[i] == digit]

        if how_many and how_many != "all":
            images = images[:how_many]
        for i in range(len(images)):
            image = images[i]
            scatterer = MNISTScatterer(image)
            scatterers.append(scatterer.get_scatterer_profile())

        filename = f"scatterers_mnist_{spec}"
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

        def _get_forward_values(scatterer):
            model = MethodOfMomentModel()
            incident_power, total_power, direct_field, total_field = \
                model.generate_forward_data(scatterer)
            return incident_power, total_power, direct_field, total_field

        results = Parallel(n_jobs=4)(delayed(_get_forward_values)(scatterer) for scatterer in scatterers)

        for j, result in enumerate(results):
            incident_powers.append(result[0])
            total_powers.append(result[1])
            incident_fields.append(result[2])
            total_fields.append(result[3])

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

    @staticmethod
    def generate_initial_guesses(scatterers, field_data):
        model = LinearInverse()
        initial_guesses = []
        scatterers = np.asarray(scatterers)
        for i in range(0, scatterers.shape[0]):
            total_forward_field = field_data["total_fields"][i]
            total_forward_power = field_data["total_powers"][i]
            incident_forward_power = field_data["incident_powers"][i]
            chi = model.get_reconstruction("prytov", total_forward_field, total_forward_power, incident_forward_power)
            initial_guesses.append(chi)
        return initial_guesses

    @staticmethod
    def save_initial_guesses(initial_guesses, spec=""):
        filename = f"initialguess_{spec}_ridge"
        np.savez(os.path.join(GenerateData.initial_guess_path, filename), initial_guesses=initial_guesses)

    @staticmethod
    def generate_all_data(digit, how_many):
        doi = Config.doi_size
        er = Config.object_permittivity
        scatterer_type = "mnist"

        data_spec = f"{digit}_{how_many}_{doi}_{er}"

        scatterers = GenerateData.generate_mnist_scatterers(digit, how_many, data_spec)
        incident_fields, incident_powers, total_fields, total_powers = GenerateData.generate_forward_data(scatterers)
        GenerateData.save_forward_data(incident_fields, incident_powers, total_fields, total_powers,
                                       scatterer_type, spec=data_spec)

        field_data = {
            "incident_fields": incident_fields,
            "total_fields": total_fields,
            "incident_powers": incident_powers,
            "total_powers": total_powers
        }
        initial_guesses = GenerateData.generate_initial_guesses(scatterers, field_data)
        GenerateData.save_initial_guesses(initial_guesses, spec=data_spec)
        return scatterers, field_data, initial_guesses


if __name__ == '__main__':

    doi = Config.doi_size
    er = Config.object_permittivity
    scatterer_type = "mnist"
    digit = 3
    how_many = "all"

    data_spec = f"test_{digit}_{how_many}_{doi}_{er}"

    scatterers = GenerateData.generate_mnist_scatterers(digit, how_many, data_spec)
    incident_fields, incident_powers, total_fields, total_powers = GenerateData.generate_forward_data(scatterers)
    GenerateData.save_forward_data(incident_fields, incident_powers, total_fields, total_powers,
                                   scatterer_type, spec=data_spec)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    filepath = os.path.join(data_dir, "scatterer_data", "scatterers_mnist_test_3_all_0.5_3.npz")
    scatterer_data = np.load(filepath)
    scatterers = scatterer_data["scatterers"]

    filepath = os.path.join(data_dir, "field_data", "forward_data_mnist_test_3_all_0.5_3.npz")
    field_data = np.load(filepath)

    initial_guesses = GenerateData.generate_initial_guesses(scatterers, field_data)
    GenerateData.save_initial_guesses(initial_guesses, spec=data_spec)
