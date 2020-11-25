import gym
from brain import SimpleRNN, Supervision

class Invaders():
    """
    docstring
    """

    env = gym.make("SpaceInvaders-v0")

    def __init__(self, **kwargs):
        self.LOAD_SUPERVISION_DATA = kwargs['load_supervision_data']
        self.RENDER_SELF_TRAIN = kwargs['render_self_train']
        self.MATCHES = kwargs['matches']
        self.NUM_OF_SUPERVISIONS = kwargs['num_of_supervisions']
        self.NUM_OF_TRAINS = kwargs['trains']
        self.SLEEP = kwargs['sleep']
        
        self.simplernn = SimpleRNN(**kwargs)
        self.supervision = Supervision(**kwargs)
    
    def run_supervision_training(self):
        if not self.LOAD_SUPERVISION_DATA:
            import time
            for m in range(self.NUM_OF_SUPERVISIONS):
                frame = self.env.reset()
                reward, action, done, info = 0, 0, False, {'ale.lives': 3}

                while (not done):
                    time.sleep(self.SLEEP)
                    self.env.render()
                    action = self.supervision.play(frame, reward, info['ale.lives'])
                    frame, reward, done, info = self.env.step(action)
                    if done:
                        self.supervision.prepare_to_save_data(m)

            self.supervision.save_supervision_data()
        else:
            self.supervision.load_npz()
            
            for m in range(self.NUM_OF_SUPERVISIONS):
                num_of_frames = self.supervision.match_buffer[m][0]
                frames = self.supervision.match_buffer[m][1]
                actions = self.supervision.match_buffer[m][2]
                rewards = self.supervision.match_buffer[m][3]
                lifes = self.supervision.match_buffer[m][4]

                self.simplernn.train(frames, rewards, lifes, actions, m, num_of_frames)

            self.simplernn.save()

        del self.supervision

    def run_self_training(self):
        for m in range(self.NUM_OF_TRAINS):
            frame = self.env.reset()
            reward, action, done, info = 0, 0, False, {'ale.lives': 3}

            while (not done):
                if self.RENDER_SELF_TRAIN or m >= self.NUM_OF_TRAINS:
                    self.env.render()
                
                self.simplernn.train(frame, reward, info['ale.lives'], action, m)
                action = self.simplernn.predict(frame)

                frame, reward, done, info = self.env.step(action)
            
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