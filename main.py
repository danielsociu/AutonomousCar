# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from gym.utils import seeding, EzPickle
import environment
import gym


def test():
    # Use a breakpoint in the code line below to debug your script.
    env = environment.CarRacing()
    env.reset()
    for _ in range(5000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
