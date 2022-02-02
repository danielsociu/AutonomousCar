import image_processing
from agent import *
import numpy as np
import random
from collections import deque


class DQN_Agent(Agent):
    def __init__(self, env, history_size=128):
        super().__init__()
        self.env = env
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

    def update_weights(self, batch_size):
        batch = random.sample(self.history, batch_size)
        train_frames = []
        train_labels = []
        for index, (frame, action_taken, reward, next_frame, solved) in enumerate(batch):
            predictions = self.model.predict(np.expand_dims(frame, axis=0))[0]
            # print(index)
            if solved:
                predictions[action_taken] = reward
            else:
                future_predictions = self.model.predict(np.expand_dims(next_frame, axis=0))[0]
                predictions[action_taken] = reward + self.gamma * np.amax(future_predictions)
            train_frames.append(frame)
            train_labels.append(predictions)
        train_frames = np.array(train_frames)
        train_labels = np.array(train_labels)
        self.model.fit(train_frames, train_labels, epochs=1, verbose=1)


    def train(self, batch_size=32, num_episodes=10000, current_frames=5, show_env=True):
        agent: Agent = Agent()
        for episode in range(num_episodes):
            state = self.env.reset()
            frame = image_processing.get_processed_image(state)

            total_reward = 0
            premature_stop = 0
            solved = False

            while True:
                if show_env:
                    self.env.render()

                action = agent.step(frame)

                accumulated_reward = 0
                for _ in range(current_frames):
                    # print(action)
                    next_state, reward, solved, _ = self.env.step(action)
                    accumulated_reward += reward
                    if solved:
                        break

                next_frame = image_processing.get_processed_image(next_state)
                self.history.append((frame, self.action_space.index(action), accumulated_reward, next_frame, solved))

                total_reward += accumulated_reward

                if len(self.history) > batch_size:
                    self.update_weights(batch_size)

                if accumulated_reward < 0:
                    premature_stop += 1
                    if premature_stop >= 25:
                        print(
                            f'Episode {episode} | Total_reward {total_reward} | Accumulated reward {accumulated_reward} | Current_epsilon {agent.epsilon}')
                        break
