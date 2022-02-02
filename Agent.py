import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self):
        # Steering, Gas, Break
        self.action_space = [
            (-1, 0.5, 0.2), (0, 0.5, 0.2), (1, 0.5, 0.2),
            (-1, 0,   0), (0, 0,   0), (1, 0,   0),
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2)
        ],

        self.LR = 1e-3
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(6, 6), activation='relu', padding="same", input_shape=(96, 96, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(6, 6), activation="relu", padding='same'))
        model.add(Dropout(0.2, seed=42))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(6, 6), activation="relu", padding='same'))
        model.add(Dropout(0.2, seed=42))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.LR))

        model.summary()  # structura modelului

        return model

    def step(self, state):
        padded_state = np.expand_dims(state, axis=0)
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(padded_state)
            print(act_values)
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randint(0, len(self.action_space))
        return self.action_space[action_index]
