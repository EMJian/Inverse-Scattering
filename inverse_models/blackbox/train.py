import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from data import Data
from model import BlackBoxModel


if __name__ == '__main__':

    # Get data
    train_input, test_input, train_output, test_output = Data.get_data()
    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)
    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)

    # Generate model
    model = BlackBoxModel.get_model()
    print(model.summary)

    # Train model
    history = model.fit(train_input,
                        train_output,
                        validation_split=0.2,
                        epochs=50,
                        batch_size=64,
                        callbacks=[EarlyStopping(monitor='val_loss')])

    # Save model
    model.save(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\PROJECTS\ISP\inverse_models\blackbox\
    trained_model\blackbox_model\14th March 2021")

    # Predict
    y_pred = model.predict(test_input)

    # View results
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
