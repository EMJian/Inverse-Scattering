import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from config import Config

from scatterers.mnist import MNISTScatterer
from forward_models.mom import MethodOfMomentModel
from utils.generate_data import GenerateData
from inverse_models.physics_assisted.data import Data


class TestInverseModel:

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    initial_guess_path = os.path.join(data_dir, "initial_guess_data")
    scatterer_path = os.path.join(data_dir, "scatterer_data")
    model = load_model(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\PROJECTS\ISP\inverse_models\physics_assisted\trained_model\3_0.5_1.5+3_ridge_v7_relu")

    @staticmethod
    def different_scatterer(digit, how_many):

        scatterers, field_data, initial_guesses = GenerateData.generate_all_data(digit, how_many)

        scatterers = Data.resize_output_data(scatterers)
        Data.check_data_sanctity(initial_guesses, scatterers)

        # Use test data
        initial_guesses = np.asarray(initial_guesses)
        reconstruction = TestInverseModel.model.predict(initial_guesses)

        for i in range(how_many):
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

            original = ax1.imshow(scatterers[i, :, :], cmap=plt.cm.gray, extent=[0, Config.doi_size, 0, Config.doi_size])
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Original scatterer")

            reconstructed = ax2.imshow(initial_guesses[i, :, :], cmap=plt.cm.gray, extent=[0, Config.doi_size, 0, Config.doi_size])
            fig.colorbar(reconstructed, ax=ax2, fraction=0.046, pad=0.04)
            ax2.title.set_text("Initial guess")

            reconstruction[i, :, :][reconstruction[i, :, :] < 0.1] = 0.85
            thresholded = ax3.imshow(reconstruction[i, :, :], cmap=plt.cm.gray, extent=[0, Config.doi_size, 0, Config.doi_size])
            fig.colorbar(thresholded, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Reconstructed scatterer")

            plt.show()

    @staticmethod
    def no_scatterer():
        scatterer = np.ones((100, 100))
        model = MethodOfMomentModel()
        sensor_positions = model.get_sensor_positions()
        receiver_positions = sensor_positions
        transmitter_positions = sensor_positions
        grid_positions = model.get_grid_positions()

        model.find_grids_with_object(grid_positions, scatterer)
        object_field = model.get_field_from_scattering(scatterer)
        direct_field = model.get_direct_field(transmitter_positions, receiver_positions)

        incident_field = model.get_incident_field(transmitter_positions, grid_positions)
        current = model.get_induced_current(object_field, incident_field)
        scattered_field = model.get_scattered_field(current, grid_positions, transmitter_positions)

        total_field = direct_field + scattered_field
        direct_field, scattered_field, total_field, txrx_pairs = model.transreceiver_manipulation(direct_field,
                                                                                                  scattered_field,
                                                                                                  total_field)

        input = total_field
        output = scatterer
        Data.check_data_sanctity(input, output)
        input = Data.split_channels(input)
        input = input[np.newaxis,:,:,:]
        input = Data.add_zero_padding(input)
        reconstruction = TestInverseModel.model.predict(input)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        original = ax1.imshow(output, cmap=plt.cm.gray)
        fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
        ax1.title.set_text("Original scatterer")

        reconstruction[0, :, :, :][reconstruction[0, :, :, :] < 1.2] = 1
        reconstructed = ax2.imshow(reconstruction[0, :, :, :], cmap=plt.cm.gray)
        fig.colorbar(reconstructed, ax=ax2, fraction=0.046, pad=0.04)
        ax2.title.set_text("Reconstructed scatterer")

        thresholded = ax3.imshow(reconstruction[0, :, :, :] > 1.5, cmap=plt.cm.gray)
        fig.colorbar(thresholded, ax=ax3, fraction=0.046, pad=0.04)
        ax3.title.set_text("Reconstructed permittivity  > 1.5")

        plt.show()

    @staticmethod
    def circular_scatterer():

        x = np.arange(0, Config.forward_grid_number)
        y = np.arange(0, Config.forward_grid_number)
        arr = np.ones((y.size, x.size))

        cx = Config.forward_grid_number/2
        cy = Config.forward_grid_number/2
        r = 15

        # The two lines below could be merged, but I stored the mask
        # for code clarity.
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
        arr[mask] = Config.object_permittivity
        arr = arr[np.newaxis, ...]
        scatterers = arr

        incident_fields, incident_powers, total_fields, total_powers = GenerateData.generate_forward_data(scatterers)

        field_data = {
            "incident_fields": incident_fields,
            "total_fields": total_fields,
            "incident_powers": incident_powers,
            "total_powers": total_powers
        }
        initial_guesses = GenerateData.generate_initial_guesses(scatterers, field_data)
        Data.check_data_sanctity(initial_guesses, scatterers)

        initial_guesses = np.asarray(initial_guesses)
        reconstruction = TestInverseModel.model.predict(initial_guesses)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        original = ax1.imshow(scatterers[0, :, :], cmap=plt.cm.gray, extent=[0, Config.doi_size, 0, Config.doi_size])
        fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
        ax1.title.set_text("Original scatterer")

        reconstructed = ax2.imshow(initial_guesses[0, :, :], cmap=plt.cm.gray,
                                   extent=[0, Config.doi_size, 0, Config.doi_size])
        fig.colorbar(reconstructed, ax=ax2, fraction=0.046, pad=0.04)
        ax2.title.set_text("Initial guess")

        thresholded = ax3.imshow(reconstruction[0, :, :], cmap=plt.cm.gray,
                                 extent=[0, Config.doi_size, 0, Config.doi_size])
        fig.colorbar(thresholded, ax=ax3, fraction=0.046, pad=0.04)
        ax3.title.set_text("Reconstructed scatterer")

        plt.show()

    @staticmethod
    def higher_permittivity():
        digit = 3
        digits = mnist.load_data()
        training_data = digits[0]
        train_X = training_data[0]
        train_Y = training_data[1]
        images_three = [x for i, x in enumerate(train_X) if train_Y[i] == digit]

        image = images_three[2]
        scatterer = MNISTScatterer(image, 100, 7)
        permittivity = scatterer.get_scatterer_profile()

        model = MethodOfMomentModel()
        sensor_positions = model.get_sensor_positions()
        receiver_positions = sensor_positions
        transmitter_positions = sensor_positions
        grid_positions = model.get_grid_positions()

        model.find_grids_with_object(grid_positions, permittivity)
        object_field = model.get_field_from_scattering(permittivity)
        direct_field = model.get_direct_field(transmitter_positions, receiver_positions)

        incident_field = model.get_incident_field(transmitter_positions, grid_positions)
        current = model.get_induced_current(object_field, incident_field)
        scattered_field = model.get_scattered_field(current, grid_positions, transmitter_positions)

        total_field = direct_field + scattered_field
        direct_field, scattered_field, total_field, txrx_pairs = model.transreceiver_manipulation(direct_field,
                                                                                                  scattered_field,
                                                                                                  total_field)

        input = total_field
        output = permittivity
        Data.check_data_sanctity(input, output)
        input = Data.split_channels(input)
        input = input[np.newaxis, :, :, :]
        input = Data.add_zero_padding(input)
        reconstruction = TestInverseModel.model.predict(input)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        original = ax1.imshow(output, cmap=plt.cm.gray)
        fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
        ax1.title.set_text("Original scatterer")

        reconstruction[0, :, :, :][reconstruction[0, :, :, :] < 1.2] = 1
        reconstructed = ax2.imshow(reconstruction[0, :, :, :], cmap=plt.cm.gray)
        fig.colorbar(reconstructed, ax=ax2, fraction=0.046, pad=0.04)
        ax2.title.set_text("Reconstructed scatterer")

        thresholded = ax3.imshow(reconstruction[0, :, :, :] > 2, cmap=plt.cm.gray)
        fig.colorbar(thresholded, ax=ax3, fraction=0.046, pad=0.04)
        ax3.title.set_text("Reconstructed permittivity  > 2")

        plt.show()

    @staticmethod
    def two_scatterers(digit, how_many):
        doi = Config.doi_size
        er = Config.object_permittivity
        data_spec = f"{digit}_{how_many}_{doi}_{er}"
        scatterers = GenerateData.generate_mnist_scatterers(digit, how_many, data_spec)

        for i, scatterer in enumerate(scatterers):
            print("i", i)
            for j in range(0, scatterer.shape[0]):
                for k in range(0, scatterer.shape[1]):
                    if scatterer[j,k] == Config.object_permittivity and j <= 40:
                        scatterer[j,k] = 1.5
                    if scatterer[j, k] == Config.object_permittivity and 40 < j <= 60:
                        scatterer[j,k] = 1
            scatterers[i] = scatterer

        incident_fields, incident_powers, total_fields, total_powers = GenerateData.generate_forward_data(scatterers)

        field_data = {
            "incident_fields": incident_fields,
            "total_fields": total_fields,
            "incident_powers": incident_powers,
            "total_powers": total_powers
        }
        initial_guesses = GenerateData.generate_initial_guesses(scatterers, field_data)
        Data.check_data_sanctity(initial_guesses, scatterers)

        # Use test data
        initial_guesses = np.asarray(initial_guesses)
        scatterers = np.asarray(scatterers)
        reconstruction = TestInverseModel.model.predict(initial_guesses)

        for i in range(0, scatterers.shape[0]):

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

            original = ax1.imshow(scatterers[i, :, :], cmap=plt.cm.gray, extent=[0, Config.doi_size, 0, Config.doi_size])
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Original scatterer")

            reconstructed = ax2.imshow(initial_guesses[i, :, :], cmap=plt.cm.gray,
                                       extent=[0, Config.doi_size, 0, Config.doi_size])
            fig.colorbar(reconstructed, ax=ax2, fraction=0.046, pad=0.04)
            ax2.title.set_text("Initial guess")

            reconstruction[i, :, :][reconstruction[i, :, :] < 0.1] = 0.85
            thresholded = ax3.imshow(reconstruction[i, :, :], cmap=plt.cm.gray,
                                     extent=[0, Config.doi_size, 0, Config.doi_size])
            fig.colorbar(thresholded, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Reconstructed scatterer")

            plt.show()


if __name__ == '__main__':

    # digit = 3
    # TestInverseModel.vary_noise_levels(digit)
    #
    # digit = 3
    # TestInverseModel.different_scatterer(digit, 10)

    # TestInverseModel.no_scatterer()

    # TestInverseModel.higher_permittivity()

    # TestInverseModel.two_scatterers(3, 5)

    TestInverseModel.circular_scatterer()

