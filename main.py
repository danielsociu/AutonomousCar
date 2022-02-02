from train import *
import gym
import environment

train_model = True
env = environment.CarRacing()
dqn_agent = DQN_Agent(env)
if train_model:
    dqn_agent.train()
