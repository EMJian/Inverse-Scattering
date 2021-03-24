import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from data import Data


if __name__ == '__main__':

    model = load_model(r"C:/Users/dsamr/OneDrive - HKUST Connect/MPHIL RESEARCH/PROJECTS/ISP/inverse_models/blackbox/trained_model/blackbox_model")
    y_pred = model.predict(test_input)

    fig = plt.figure(figsize=(8, 8))

    image = fig.add_subplot(2, 3, 1)
    plt.imshow(test_output[8, :, :], cmap=plt.cm.gray)
    image.title.set_text("Original scatterer")

    binary = fig.add_subplot(2, 3, 2)
    plt.imshow(y_pred[8, :, :, :], cmap=plt.cm.gray)
    binary.title.set_text("Reconstructed")

    binary = fig.add_subplot(2, 3, 3)
    plt.imshow(y_pred[8, :, :, :] > 2, cmap=plt.cm.gray)
    binary.title.set_text("Reconstructed permittivity > 2")

    image = fig.add_subplot(2, 3, 4)
    plt.imshow(test_output[15, :, :], cmap=plt.cm.gray)
    image.title.set_text("Original scatterer")

    binary = fig.add_subplot(2, 3, 5)
    plt.imshow(y_pred[15, :, :, :], cmap=plt.cm.gray)
    binary.title.set_text("Reconstructed")

    binary = fig.add_subplot(2, 3, 6)
    plt.imshow(y_pred[15, :, :, :] > 2, cmap=plt.cm.gray)
    binary.title.set_text("Reconstructed permittivity > 2")

    plt.show()
