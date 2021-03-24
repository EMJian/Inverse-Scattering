from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from scatterers.mnist import MNISTScatterer
from data import Data
from forward_models.mom import MethodOfMomentModel


class TestInverseModel:

    model = load_model(
        r"C:/Users/dsamr/OneDrive - HKUST Connect/MPHIL RESEARCH/PROJECTS/ISP/inverse_models/blackbox/trained_model/blackbox_model")

    @staticmethod
    def vary_noise_levels(digit):
        train_input, test_input, train_output, test_output = Data.get_data(digit, test_size=9)

        noise_levels = [0, 0.01]
        for j, noise_level in enumerate(noise_levels):

            noise = np.random.normal(0, noise_level, test_input.shape)
            new_input = test_input + noise

            reconstruction = TestInverseModel.model.predict(new_input)

            i = 3

            fig = plt.figure(figsize=(8, 8))

            image = fig.add_subplot(3, 3, 3*j+1)
            plt.imshow(test_output[i, :, :], cmap=plt.cm.gray)
            image.title.set_text("Original scatterer")

            binary = fig.add_subplot(3, 3, 3*j+2)
            plt.imshow(reconstruction[i, :, :, :], cmap=plt.cm.gray)
            binary.title.set_text(f"Noise variance {noise_level}")

            binary = fig.add_subplot(3, 3, 3*j+3)
            plt.imshow(reconstruction[i, :, :, :] > 2, cmap=plt.cm.gray)
            binary.title.set_text("Permittivity > 2")

        plt.show()

    @staticmethod
    def different_scatterer(digit):
        train_input, test_input, train_output, test_output = Data.get_data(digit, test_size=9)

        # Use test data
        reconstruction = TestInverseModel.model.predict(test_input)

        for i in range(9):
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

            original = ax1.imshow(test_output[i, :, :], cmap=plt.cm.gray)
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Original scatterer")

            reconstruction[i, :, :, :][reconstruction[i, :, :, :] < 1.2] = 1
            reconstructed = ax2.imshow(reconstruction[i, :, :, :], cmap=plt.cm.gray)
            fig.colorbar(reconstructed, ax=ax2, fraction=0.046, pad=0.04)
            ax2.title.set_text("Reconstructed scatterer")

            thresholded = ax3.imshow(reconstruction[i, :, :, :] > 2, cmap=plt.cm.gray)
            fig.colorbar(thresholded, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Reconstructed permittivity  > 2")

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

        x = np.arange(0, 100)
        y = np.arange(0, 100)
        arr = np.ones((y.size, x.size))

        cx = 50
        cy = 50
        r = 30

        # The two lines below could be merged, but I stored the mask
        # for code clarity.
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
        arr[mask] = 3

        scatterer = arr

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
        input = input[np.newaxis, :, :, :]
        input = Data.add_zero_padding(input)
        reconstruction = TestInverseModel.model.predict(input)

        fig = plt.figure(figsize=(8, 8))

        image = fig.add_subplot(2, 3, 1)
        plt.imshow(output, cmap=plt.cm.gray)
        image.title.set_text("Original scatterer")

        binary = fig.add_subplot(2, 3, 2)
        plt.imshow((reconstruction[0, :, :, :] > 1.2)*reconstruction[0, :, :, :], cmap=plt.cm.gray)
        binary.title.set_text("Reconstructed")

        binary = fig.add_subplot(2, 3, 3)
        plt.imshow(reconstruction[0, :, :, :] > 2, cmap=plt.cm.gray)
        binary.title.set_text("Reconstructed permittivity > 2")

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


if __name__ == '__main__':

    # digit = 3
    # TestInverseModel.vary_noise_levels(digit)
    #
    digit = 3
    TestInverseModel.different_scatterer(digit)

    # TestInverseModel.no_scatterer()

    # TestInverseModel.higher_permittivity()

    # TestInverseModel.circular_scatterer()
