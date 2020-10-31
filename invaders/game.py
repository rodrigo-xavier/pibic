import gym
from brain import SimpleRNN

class Invaders():
    """
    docstring
    """

    env = gym.make("SpaceInvaders-v0")

    def __init__(self, path, match):
        self.path = path
        self.match = match
        self.lstm = SimpleRNN(path=path)
    
    def run(self):
        for m in range(self.match):
            frame = self.env.reset()
            reward = 0
            action = 0
            done = False

            while (not done):
                # if m > 20:
                self.env.render()
                action = self.lstm.predict(frame, reward)
                frame, reward, done, info = self.env.step(action)

        self.lstm.save()
        self.env.close()