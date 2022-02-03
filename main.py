from train import *
import gym
import environment

train_model = False
load_model = True
model_used = "./logs/branch_930_runs/model_270_4.h5"
episodes = 50
run_nr = 5
env = environment.CarRacing()
dqn_agent = DQN_Agent(env)
if train_model:
    if load_model:
        dqn_agent.train(model_used=model_used, run_nr=run_nr)
    else:
        dqn_agent.train(run_nr=run_nr)
else:
    dqn_agent.play_model(model_used, episodes)
