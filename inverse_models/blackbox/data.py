import os
import numpy as np
from sklearn.model_selection import train_test_split


class Data:

    @staticmethod
    def read_data():
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_file = os.path.join(script_dir, "data", "field_data", "mnist_total_field_3s.npz")
        output_file = os.path.join(script_dir, "data", "scatterer_data", "mnist_scatterers_3.npz")

        input = np.load(input_file)
        input = input["total_field_3s"]

        output = np.load(output_file)
        output = output["mnist_scatterers_3"]

        print("Input data dimensions: ", input.shape)
        print("Output data dimensions: ", output.shape)

        return input, output

    @staticmethod
    def split_data(input, output):
        train_input, train_output, test_input, test_output = train_test_split(input, output, test_size=0.1, random_state=42)
        return train_input, train_output, test_input, test_output

    @staticmethod
    def get_data():
        input, output = Data.read_data()
        train_input, test_input, train_output, test_output = Data.split_data(input, output)
        return train_input, test_input, train_output, test_output


if __name__ == '__main__':

    train_input, test_input, train_output, test_output = Data.get_data()
    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)
    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)
