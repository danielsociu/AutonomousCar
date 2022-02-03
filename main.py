from train import *
import gym
import environment

train_model = True
model_used = "./logs/model_168.h5"
episodes = 50
env = environment.CarRacing()
dqn_agent = DQN_Agent(env)
if train_model:
    dqn_agent.train()
else:
    dqn_agent.play_model(model_used, episodes)
