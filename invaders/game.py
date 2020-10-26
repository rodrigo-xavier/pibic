import gym
from brain import LSTM

class Invaders():
    """
    docstring
    """

    env = gym.make("SpaceInvaders-v0")

    def __init__(self, path, match):
        self.path = path
        self.match = match
        self.lstm = LSTM()
    
    def run(self):
        for m in range(self.match):
            observation = self.env.reset()
            reward = 0
            done = False

            while (not done):
                self.env.render()
                action = self.lstm.train(observation, reward)
                observation, reward, done, info = self.env.step(action)


        self.env.close()