import image_processing
from Agent import *
import numpy as np
from collections import deque


def train(env, batch_size=256, num_episodes=10000, current_frames=5, show_env=True):
    agent: Agent = Agent()
    history = []
    for episode in range(num_episodes):
        state = env.reset()
        frame = image_processing.get_processed_image(state)

        total_reward = 0
        premature_stop = 0
        solved = False

        while True:
            if show_env:
                env.render()

            action = agent.step(frame)

            accumulated_reward = 0
            for _ in range(current_frames):
                # print(action)
                next_state, reward, solved, _ = env.step(action)
                accumulated_reward += reward
                if solved:
                    break

            next_frame = image_processing.get_processed_image(next_state)
            history.append((frame, action, accumulated_reward, next_frame, solved))

            total_reward += accumulated_reward

            if accumulated_reward < 0:
                premature_stop += 1
                if premature_stop >= 100:
                    print(f'Episode {episode} | Total_reward {total_reward} | Accumulated reward {accumulated_reward} | Current_epsilon {agent.epsilon}')
                    break

    env.close()







