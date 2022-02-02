import image_processing
from agent import *
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def plt_metric(history, metric, title, nr, has_valid=True):
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.savefig("logs/customModel-5_" + title + "_" + nr + ".png", bbox_inches='tight', dpi=300)
    plt.show()

class DQN_Agent(Agent):
    def __init__(self, env, history_size=128):
        super().__init__()
        self.env = env
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

    def update_weights(self, batch_size, episode):
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
        fit_hist = self.model.fit(train_frames, train_labels, epochs=1, verbose=1)
        plt_metric(history=fit_hist.history, metric="loss", title="mean_squared_error", nr=episode)

    def train(self, batch_size=32, num_episodes=10000, current_frames=5, save_frequency=5, show_env=True):
        writer_logdir = 'logs'
        writer = SummaryWriter(log_dir=writer_logdir)
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
                    self.update_weights(batch_size, episode)

                if accumulated_reward < 0:
                    premature_stop += 1
                    if premature_stop >= 25:
                        print(
                            f'Episode {episode} | Total_reward {total_reward} | Accumulated_reward {accumulated_reward} | Current_epsilon {agent.epsilon}')
                        writer.add_scalar(tag='Total_reward',
                                          scalar_value=total_reward,
                                          global_step=episode)
                        writer.add_scalar(tag='Accumulated_reward',
                                          scalar_value=accumulated_reward,
                                          global_step=episode)
                        writer.add_scalar(tag='Current_epsilon',
                                          scalar_value=agent.epsilon,
                                          global_step=episode)
                        writer.flush()
                        break

            if episode % save_frequency == 0:
                agent.save(writer_logdir + "/model_" + str(episode)+".h5")
