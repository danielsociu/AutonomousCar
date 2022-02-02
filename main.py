from train import *
import gym

train_model = True
env = gym.make('CarRacing-v0')
if train_model:
    train(env)
