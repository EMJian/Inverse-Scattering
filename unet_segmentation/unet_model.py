from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


class UNetSegmentation:

    @staticmethod
    def get_model(pretrained_weights=None):

        input_layer = Input(shape=(512, 512, 1))

        # Down-sampling
        conv1 = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(input_layer)
        conv1 = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(conv1)
        pool1 = MaxPooling2D(pool_size=2)(conv1)

        conv2 = Conv2D(128, kernel_size=3, padding="SAME", activation="relu")(pool1)
        conv2 = Conv2D(128, kernel_size=3, padding="SAME", activation="relu")(conv2)
        pool2 = MaxPooling2D(pool_size=2)(conv2)

        conv3 = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(pool2)
        conv3 = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(conv3)
        pool3 = MaxPooling2D(pool_size=2)(conv3)

        conv4 = Conv2D(512, kernel_size=3, padding="SAME", activation="relu")(pool3)
        conv4 = Conv2D(512, kernel_size=3, padding="SAME", activation="relu")(conv4)
        pool4 = MaxPooling2D(pool_size=2)(conv4)

        conv5 = Conv2D(1024, kernel_size=3, padding="SAME", activation="relu")(pool4)
        conv5 = Conv2D(1024, kernel_size=3, padding="SAME", activation="relu")(conv5)

        # Up-sampling
        up6 = (UpSampling2D(size=(2, 2))(conv5))
        up6 = Conv2D(512, kernel_size=2, padding="SAME", activation='relu')(up6)
        merge6 = Concatenate()([conv4, up6])
        conv6 = Conv2D(512, kernel_size=3, padding="SAME", activation="relu")(merge6)
        conv6 = Conv2D(512, kernel_size=3, padding="SAME", activation="relu")(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(up7)
        merge7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(merge7)
        conv7 = Conv2D(256, kernel_size=3, padding="SAME", activation="relu")(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = Conv2D(128, kernel_size=3, padding="SAME", activation="relu")(up8)
        merge8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(128, kernel_size=3, padding="SAME", activation="relu")(merge8)
        conv8 = Conv2D(128, kernel_size=3, padding="SAME", activation="relu")(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(up9)
        merge9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(merge9)
        conv9 = Conv2D(64, kernel_size=3, padding="SAME", activation="relu")(conv9)

        conv10 = Conv2D(1, kernel_size=1, activation="sigmoid")(conv9)

        model = Model(inputs=input_layer, outputs=conv10)
        model.compile(optimizer=Adam(learning_rate=1e-5), loss=BinaryCrossentropy, metrics=["accuracy"])
        return model
