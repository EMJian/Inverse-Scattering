import os
import numpy as np
from sklearn.model_selection import train_test_split


class Data:

    @staticmethod
    def read_data(digit):
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_file = os.path.join(script_dir, "data", "field_data", f"mnist_total_field_{digit}s.npz")
        output_file = os.path.join(script_dir, "data", "scatterer_data", f"mnist_scatterers_{digit}.npz")

        input = np.load(input_file)
        input = input[f"total_field_{digit}s"]

        output = np.load(output_file)
        output = output[f"mnist_scatterers_{digit}"]

        print("Input data dimensions: ", input.shape)
        print("Output data dimensions: ", output.shape)

        return input, output

    @staticmethod
    def check_data_sanctity(input, output):
        assert not np.isnan(input).any()
        assert not np.isnan(output).any()

    @staticmethod
    def split_channels(input):
        input = np.asarray([input.real, input.imag])
        input = np.moveaxis(input, 0, -1)
        return input

    @staticmethod
    def add_zero_padding(input):
        input = np.pad(input, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="constant")
        return input

    @staticmethod
    def split_data(input, output, test_size=0.1):
        train_input, train_output, test_input, test_output = train_test_split(input, output, test_size=test_size, random_state=42)
        return train_input, train_output, test_input, test_output

    @staticmethod
    def get_data(digit, test_size=0.1):
        input, output = Data.read_data(digit)
        Data.check_data_sanctity(input, output)
        input = Data.split_channels(input)
        input = Data.add_zero_padding(input)
        train_input, test_input, train_output, test_output = Data.split_data(input, output, test_size=test_size)
        return train_input, test_input, train_output, test_output


if __name__ == '__main__':

    digit = 3
    train_input, test_input, train_output, test_output = Data.get_data(digit)
    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)
    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)
