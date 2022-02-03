from train import *
import gym
import environment

train_model = False
model_used = "./logs/model_144_2.h5"
episodes = 50
run_nr = 2
env = environment.CarRacing()
dqn_agent = DQN_Agent(env)
if train_model:
    dqn_agent.train(run_nr=run_nr)
else:
    dqn_agent.play_model(model_used, episodes)
