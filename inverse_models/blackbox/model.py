from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.losses import MeanSquaredError


class BlackBoxModel:

    @staticmethod
    def get_model():

        model = Sequential([
            Conv2D(16, kernel_size=3, padding="SAME", input_shape=(40, 40, 2), data_format="channels_last"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=2), # 20

            Conv2D(32, kernel_size=3, padding="SAME"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=2), # 10

            Conv2D(64, kernel_size=3, padding="SAME"),
            BatchNormalization(),
            Activation("relu"),
            UpSampling2D(size=(2, 2)), # 20

            Conv2D(64, kernel_size=3, padding="SAME"),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(32, kernel_size=3, padding="SAME"),
            BatchNormalization(),
            Activation("relu"),

            UpSampling2D(size=(5, 5)),  # 100
            Conv2D(16, kernel_size=3, padding="SAME"),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(1, kernel_size=1, activation="relu")
        ])

        model.summary()

        model.compile(
            optimizer="adam",
            loss=MeanSquaredError(),
            metrics=["accuracy", "mse"]
        )

        return model


if __name__ == '__main__':

    model = BlackBoxModel.get_model()
    print(model.summary)
