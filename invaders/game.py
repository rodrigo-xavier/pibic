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
        self.simplernn = SimpleRNN(path=path)
    
    def run(self):
        for m in range(self.match):
            frame = self.env.reset()

            info = {'ale.lives': 3}
            reward = 0
            action = 0
            done = False

            while (not done):
                # if m >= 10:
                self.env.render()

                action = self.simplernn.predict(frame, reward, info, m)
                frame, reward, done, info = self.env.step(action)
            
        self.simplernn.save()
        self.env.close()