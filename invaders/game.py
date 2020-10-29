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
        self.lstm = LSTM(path=path)
    
    def run(self):
        for m in range(self.match):
            observation = self.env.reset()
            reward = 0
            action = 0
            done = False

            while (not done):
                self.env.render()
                action = self.lstm.train(observation, reward, action)
                observation, _reward, done, info = self.env.step(action)
                reward += _reward

        self.lstm.save()
        self.env.close()