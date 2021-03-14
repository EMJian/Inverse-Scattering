from data import Data
from model import BlackBoxModel

if __name__ == '__main__':

    train_input, test_input, train_output, test_output = Data.get_data()
    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)
    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)

    model = BlackBoxModel.get_model()
    print(model.summary)

    history = model.fit(train_input, train_output, validation_split=0.2, epochs=10, batch_size=64)

    model.save(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\PROJECTS\ISP\inverse_models\blackbox\
    trained_model\blackbox_model\14th March 2021")
