import os
import numpy as np
from skimage.transform import resize

from config import Config


class Data:

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    initial_guess_path = os.path.join(data_dir, "initial_guess_data")
    scatterer_path = os.path.join(data_dir, "scatterer_data")

    @staticmethod
    def check_data_sanctity(input, output):
        assert not np.isnan(input).any()
        assert not np.isnan(output).any()

    @staticmethod
    def get_input_data(filename):
        filepath = os.path.join(Data.initial_guess_path, filename)
        initial_data = np.load(filepath)
        initial_guesses = initial_data["initial_guesses"]
        return initial_guesses

    @staticmethod
    def get_output_data(filename):
        filepath = os.path.join(Data.scatterer_path, filename)
        scatterer_data = np.load(filepath)
        scatterers = scatterer_data["scatterers"]
        return scatterers

    @staticmethod
    def resize_output_data(scatterers):
        resized_scatterers = []
        for scatterer in scatterers:
            resized = resize(scatterer, (Config.inverse_grid_number, Config.inverse_grid_number))
            resized_scatterers.append(resized)
        resized_scatterers = np.asarray(resized_scatterers)
        return resized_scatterers

    @staticmethod
    def split_data(input, output, test_size=0.1):
        test_data_len = int(input.shape[0] * test_size)
        train_data_len = input.shape[0] - test_data_len
        train_input, train_output = input[:train_data_len, :, :], output[:train_data_len, :, :]
        test_input, test_output = input[train_data_len:, :, :], output[train_data_len:, :, :]
        return train_input, train_output, test_input, test_output

    @staticmethod
    def get_data(input_file, output_file):
        X = Data.get_input_data(input_file)
        y = Data.get_output_data(output_file)
        y = Data.resize_output_data(y)
        Data.check_data_sanctity(X, y)
        train_input, train_output, test_input, test_output = Data.split_data(X, y)
        return train_input, train_output, test_input, test_output


if __name__ == '__main__':

    input_file = "initialguess_3_all_0.5_1.5_ridge.npz"
    output_file = "scatterers_mnist_3_all_0.5_1.5.npz"
    train_input, train_output, test_input, test_output = Data.get_data(input_file, output_file)
    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)
    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)
