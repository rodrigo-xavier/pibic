import gym
from brain import SimpleRNN

class Invaders():
    """
    docstring
    """

    env = gym.make("SpaceInvaders-v0")

    def __init__(self, **kwargs):
        self.MATCHES = kwargs['matches']
        self.NUM_OF_TRAINS = kwargs['trains']
        self.RENDER = kwargs['render']
        self.simplernn = SimpleRNN(**kwargs)
    
    def run(self):
        for m in range(self.MATCHES):
            frame = self.env.reset()

            info = {'ale.lives': 3}
            reward = 0
            action = 0
            done = False

            while (not done):
                if self.RENDER or m >= self.NUM_OF_TRAINS:
                    self.env.render()

                action = self.simplernn.predict(frame, reward, info, m)
                frame, reward, done, info = self.env.step(action)
            
        self.simplernn.save()
        self.env.close()