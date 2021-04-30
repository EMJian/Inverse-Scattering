from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, BatchNormalization, Activation, Dropout, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class PhysicsAssistedModel:

    @staticmethod
    def get_model(pretrained_weights=None):

        input_layer = Input(shape=(50, 50, 1))

        # Down-sampling
        conv1 = Conv2D(64, kernel_size=3, padding="VALID")(input_layer) # 48 x 48
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        # conv1 = Dropout(0.1)(conv1)

        conv1 = Conv2D(64, kernel_size=3, padding="SAME")(conv1)  # 48 x 48
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        # conv1 = Dropout(0.1)(conv1)

        conv1 = Conv2D(64, kernel_size=3, padding="SAME")(conv1)  # 48 x 48
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        # conv1 = Dropout(0.1)(conv1)

        pool1 = MaxPooling2D(pool_size=2)(conv1)  # 24 x 24

        conv2 = Conv2D(128, kernel_size=3, padding="SAME")(pool1) # 24 x 24
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        # conv2 = Dropout(0.1)(conv2)

        conv2 = Conv2D(128, kernel_size=3, padding="SAME")(conv2) # 24 x 24
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        # conv2 = Dropout(0.1)(conv2)

        conv2 = Conv2D(128, kernel_size=3, padding="SAME")(conv2) # 24 x 24
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        # conv2 = Dropout(0.1)(conv2)

        pool2 = MaxPooling2D(pool_size=2)(conv2)  # 12 x 12

        conv3 = Conv2D(256, kernel_size=3, padding="SAME")(pool2)  # 12 x 12
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        # conv3 = Dropout(0.1)(conv3)

        conv3 = Conv2D(256, kernel_size=3, padding="SAME")(conv3)  # 12 x 12
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        # conv3 = Dropout(0.1)(conv3)

        conv3 = Conv2D(256, kernel_size=3, padding="SAME")(conv3)  # 12 x 12
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        # conv3 = Dropout(0.1)(conv3)

        pool3 = MaxPooling2D(pool_size=2)(conv3)  # 6 x 6

        conv4 = Conv2D(512, kernel_size=3, padding="SAME")(pool3)  # 6 x 6
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        # conv4 = Dropout(0.1)(conv4)

        conv4 = Conv2D(512, kernel_size=3, padding="SAME")(conv4)  # 6 x 6
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        # conv4 = Dropout(0.1)(conv4)

        conv4 = Conv2D(512, kernel_size=3, padding="SAME")(conv4)  # 6 x 6
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        # conv4 = Dropout(0.1)(conv4)

        ###

        # Up-sampling
        up5 = (UpSampling2D(size=(2, 2))(conv4))  # 12 x 12
        merge5 = Concatenate()([conv3, up5])

        conv5 = Conv2D(256, kernel_size=2, padding="SAME")(merge5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        # conv5 = Dropout(0.1)(conv5)

        conv5 = Conv2D(256, kernel_size=3, padding="SAME")(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        # conv5 = Dropout(0.1)(conv5)

        conv5 = Conv2D(256, kernel_size=3, padding="SAME")(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        # conv5 = Dropout(0.1)(conv5)

        up6 = (UpSampling2D(size=(2, 2))(conv5))  # 24 x 24
        merge6 = Concatenate()([conv2, up6])

        conv6 = Conv2D(128, kernel_size=2, padding="SAME")(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        # conv6 = Dropout(0.1)(conv6)

        conv6 = Conv2D(128, kernel_size=3, padding="SAME")(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        # conv6 = Dropout(0.1)(conv6)

        conv6 = Conv2D(128, kernel_size=3, padding="SAME")(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        # conv6 = Dropout(0.1)(conv6)

        up7 = (UpSampling2D(size=(2, 2))(conv6))  # 48 x 48
        merge7 = Concatenate()([conv1, up7])

        conv7 = Conv2D(64, kernel_size=2, padding="SAME")(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        # conv7 = Dropout(0.1)(conv7)

        conv7 = Conv2D(64, kernel_size=3, padding="SAME")(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        # conv7 = Dropout(0.1)(conv7)

        conv7 = Conv2D(64, kernel_size=3, padding="SAME")(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        # conv7 = Dropout(0.1)(conv7)

        conv8 = Conv2DTranspose(1, kernel_size=3, padding="VALID")(conv7) # 50 x 50

        merge9 = Concatenate()([conv8, input_layer])

        conv10 = Conv2D(1, kernel_size=1)(merge9)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation("relu")(conv10)

        model = Model(inputs=input_layer, outputs=conv10)

        initial_learning_rate = 0.1
        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=MeanSquaredError(), metrics=["accuracy"])
        return model


if __name__ == '__main__':

    model = PhysicsAssistedModel.get_model()
    print(model.summary())
