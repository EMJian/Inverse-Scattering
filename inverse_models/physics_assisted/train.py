import os
import random
import numpy as np
import matplotlib.pyplot as plt

from inverse_models.physics_assisted.data import Data
from inverse_models.physics_assisted.model import PhysicsAssistedModel


if __name__ == '__main__':

    # Model saving details
    model_name = "3_0.5_1.5+3_ridge_v8"
    model_dir = os.path.join(os.path.dirname(__file__), "trained_model")
    folder_name = os.path.join(model_dir, model_name)

    # Generate model
    model = PhysicsAssistedModel.get_model()
    print(model.summary)

    # Get training data
    input_file = "initialguess_3_all_0.5_1.5_ridge.npz"
    output_file = "scatterers_mnist_3_all_0.5_1.5.npz"
    train_input_2, train_output_2, test_input_2, test_output_2 = Data.get_data(input_file, output_file)

    input_file = "initialguess_3_all_0.5_3_ridge.npz"
    output_file = "scatterers_mnist_3_all_0.5_3.npz"
    train_input_1, train_output_1, test_input_1, test_output_1 = Data.get_data(input_file, output_file)

    input_file = "initialguess_test_3_all_0.5_3_ridge.npz"
    output_file = "scatterers_mnist_test_3_all_0.5_3.npz"
    train_input_3, train_output_3, test_input_3, test_output_3 = Data.get_data(input_file, output_file)

    input_file = "initialguess_test_3_all_0.5_1.5_ridge.npz"
    output_file = "scatterers_mnist_test_3_all_0.5_1.5.npz"
    train_input_4, train_output_4, test_input_4, test_output_4 = Data.get_data(input_file, output_file)

    train_input = np.concatenate((train_input_1, train_input_2, train_input_3, train_input_4), axis=0)
    train_output = np.concatenate((train_output_1, train_output_2, train_output_3, train_output_4), axis=0)
    test_input = np.concatenate((test_input_1, test_input_2, test_input_3, test_input_4), axis=0)
    test_output = np.concatenate((test_output_1, test_output_2, test_output_3, test_output_4), axis=0)

    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)
    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)

    # Train model
    history = model.fit(train_input,
                        train_output,
                        shuffle=True,
                        validation_split=0.1,
                        epochs=5,
                        batch_size=32)

    # Save model
    model.save(folder_name)

    # Predict
    y_pred = model.predict(test_input)

    # View results
    for i in random.sample(range(0, test_input.shape[2]), 5):

        fig = plt.figure(figsize=(8, 8))

        image = fig.add_subplot(1, 3, 1)
        plt.imshow(test_output[i, :, :], cmap=plt.cm.gray)
        image.title.set_text("Original scatterer")

        binary = fig.add_subplot(1, 3, 2)
        plt.imshow(y_pred[i, :, :, :], cmap=plt.cm.gray)
        binary.title.set_text("Reconstructed")

        binary = fig.add_subplot(1, 3, 3)
        plt.imshow(test_input[i, :, :], cmap=plt.cm.gray)
        binary.title.set_text("Input")

        plt.show()
