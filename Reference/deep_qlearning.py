from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


class DQNAgent:
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALULES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(Loss="mse", optimizer=Adam(Lr=0.001), metrics=["accuracy"])

