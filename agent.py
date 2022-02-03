import random

import cv2
import numpy as np
from collections import deque
import image_processing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, env, current_frames):
        # Steering, Gas, Break
        self.action_space = [
            (-1, 0.5, 0.2), (0, 0.5, 0.2), (1, 0.5, 0.2),
            (-1, 0.5, 0), (0, 0.5, 0), (1, 0.5, 0),
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
            (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        ]
        self.env = env
        self.shape = (84, 96)
        self.LR = 1e-3
        self.current_frames = current_frames
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.temporary_model = self.build_model(self.shape)
        self.model = self.build_model(self.shape)

    def update_actual_weights(self):
        self.model.set_weights(self.temporary_model.get_weights())

    def play_model(self, path, num_episodes=5):
        self.load(path)
        for e in range(num_episodes):
            state = self.env.reset()
            frame = image_processing.get_processed_image(state)

            while True:
                self.env.render()

                action = self.step(frame)
                # cv2.imshow('test', frame)
                # cv2.waitKey(0)

                for _ in range(self.current_frames):
                    # print(action)
                    next_state, reward, solved, _ = self.env.step(action)
                    if solved:
                        break
                # next_state, reward, solved, _ = self.env.step(action)

                next_frame = image_processing.get_processed_image(next_state)
                frame = next_frame

                if solved:
                    print ('solved')
                    break

    def build_model(self, shape):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(7, 7), activation='relu', strides=3, input_shape=(shape[0], shape[1], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="relu"))
        model.add(Dropout(0.2, seed=42))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.LR))

        model.summary()  # structura modelului

        return model

    def step(self, state):
        padded_state = np.expand_dims(state, axis=0)
        if np.random.rand() > self.epsilon:
            act_values = self.temporary_model.predict(padded_state)
            # print(act_values)
            action_index = np.argmax(act_values[0])
        else:
            action_index = np.random.randint(0, len(self.action_space))
            # print(self.action_space)
            # print(len(self.action_space))
            # print(action_index)
        return self.action_space[action_index]

    def save(self, path):
        self.model.save(path, save_format='h5')

    def load(self, path):
        self.temporary_model = keras.models.load_model(path)
        self.update_actual_weights()

