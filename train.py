import image_processing
from agent import *
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def plt_metric(history, metric, episode, nr):
    plt.plot(history)
    plt.title(episode)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.savefig("logs/customModel-5_" + episode + "_" + nr + ".png", bbox_inches='tight', dpi=300)
    plt.show()

class DQN_Agent(Agent):
    def __init__(self, env, history_size=512, current_frames=3):
        super().__init__(env, current_frames)
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

    def update_weights(self, batch_size, episode):
        batch = random.sample(self.history, batch_size)
        train_frames = []
        train_labels = []
        all_frames = np.array([data[0] for data in batch])
        all_next_frames = np.array([data[3] for data in batch])
        all_predictions = self.temporary_model.predict(all_frames)
        all_future_predictions = self.temporary_model.predict(all_next_frames)
        # print(all_next_frames.shape)
        # print(all_predictions)
        # print(all_future_predictions.shape)
        for index, (frame, action_taken, reward, next_frame, solved) in enumerate(batch):
            # predictions = self.temporary_model.predict(np.expand_dims(frame, axis=0))[0]
            predictions = all_predictions[index]
            if solved:
                predictions[action_taken] = reward
            else:
                # future_predictions = self.model.predict(np.expand_dims(next_frame, axis=0))[0]
                future_predictions = all_future_predictions[index]
                predictions[action_taken] = reward + self.gamma * np.amax(future_predictions)
            train_frames.append(frame)
            train_labels.append(predictions)
        train_frames = np.array(train_frames)
        train_labels = np.array(train_labels)
        temporary_model_history = self.temporary_model.fit(train_frames, train_labels, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return temporary_model_history.history['loss']

    def train(self, batch_size=64, num_episodes=10000, save_frequency=6, update_frequency=3, show_env=True, run_nr=1):
        writer_logdir = 'logs'
        writer = SummaryWriter(log_dir=writer_logdir)
        tensorboard_index = 0
        for episode in range(num_episodes):
            arr_temporary_model_history = []
            state = self.env.reset()
            frame = image_processing.get_processed_image(state)

            total_reward = 0
            premature_stop = 0
            solved = False
            temporary_model_history = None
            while True:
                if show_env:
                    self.env.render()

                action = self.step(frame)

                accumulated_reward = 0
                for _ in range(self.current_frames):
                    # print(action)
                    next_state, reward, solved, _ = self.env.step(action)
                    accumulated_reward += reward
                    if solved:
                        break

                next_frame = image_processing.get_processed_image(next_state)
                self.history.append((frame, self.action_space.index(action), accumulated_reward, next_frame, solved))

                total_reward += accumulated_reward

                if len(self.history) > batch_size:
                    temporary_model_history = self.update_weights(batch_size, episode)
                    arr_temporary_model_history.append(temporary_model_history)

                frame = next_frame
                # cv2.imshow('test', frame)
                # cv2.waitKey(0)
                tensorboard_index += 1
                writer.add_scalar(tag='Accumulated_reward',
                                  scalar_value=accumulated_reward,
                                  global_step=tensorboard_index)
                writer.add_scalar(tag='Current_epsilon',
                                  scalar_value=self.epsilon,
                                  global_step=tensorboard_index)
                writer.flush()
                if accumulated_reward < 0:
                    premature_stop += 1
                    if premature_stop >= 100:
                        print(
                            f'Episode {episode} | Total_reward {total_reward} | Accumulated_reward {accumulated_reward} | Current_epsilon {self.epsilon}')
                        writer.add_scalar(tag='Total_reward',
                                          scalar_value=total_reward,
                                          global_step=episode)
                        writer.flush()
                        break

            if episode % update_frequency == 0:
                self.update_actual_weights()
                if temporary_model_history is not None:
                    plt_metric(arr_temporary_model_history, "loss", str(episode), str(run_nr))

            if episode % save_frequency == 0:
                self.save(writer_logdir + "/model_" + str(episode) + "_" + str(run_nr) + ".h5")
