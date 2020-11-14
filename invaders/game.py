import gym
from brain import SimpleRNN, Supervision

class Invaders():
    """
    docstring
    """

    env = gym.make("SpaceInvaders-v0")

    def __init__(self, **kwargs):
        self.LOAD_SUPERVISION_DATA = kwargs['load_supervision_data']
        self.RENDER_TRAIN = kwargs['render_train']
        self.MATCHES = kwargs['matches']
        self.NUM_OF_SUPERVISIONS = kwargs['num_of_supervisions']
        self.NUM_OF_TRAINS = kwargs['trains']
        
        self.simplernn = SimpleRNN(**kwargs)
        self.supervision = Supervision(**kwargs)
    
    def run_supervision_training(self):
        if not self.LOAD_SUPERVISION_DATA:
            for m in range(self.NUM_OF_SUPERVISIONS):
                frame = self.env.reset()

                reward, action, done, info = 0, 0, False, {'ale.lives': 3}

                while (not done):
                    action = self.supervision.play(frame, reward, info['ale.lives'], m)
                    frame, reward, done, info = self.env.step(action)

            self.env.close()
            self.supervision.save_supervision_data()
        else:
            self.supervision.load_supervision_data()
            
        for m in range(len(self.supervision.match_buffer)):
            for frame in self.supervision.match_buffer[m][0]:
                self.simplernn.train(frame, )

        self.simplernn.save()

        del self.supervision

    def run_self_training(self):
        for m in range(self.NUM_OF_TRAINS):
            frame = self.env.reset()

            reward, action, done, info = 0, 0, False, {'ale.lives': 3}

            while (not done):
                if self.RENDER_TRAIN or m >= self.NUM_OF_TRAINS:
                    self.env.render()
                
                self.simplernn.train(frame, reward, info['ale.lives'], action)
                action = self.simplernn.predict(frame)

                frame, reward, done, info = self.env.step(action)
            
        self.env.close()
        self.simplernn.save()
        
    def run_predict(self):
        for m in range(self.MATCHES):
            frame = self.env.reset()

            reward, action, done, info = 0, 0, False, {'ale.lives': 3}

            while (not done):
                self.env.render()
                action = self.simplernn.predict(frame)
                frame, reward, done, info = self.env.step(action)
            
        self.env.close()