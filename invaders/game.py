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
        self.SLEEP = kwargs['sleep']
        
        self.simplernn = SimpleRNN(**kwargs)
        self.supervision = Supervision(**kwargs)
    
    def run_supervision_training(self):
        import time

        if not self.LOAD_SUPERVISION_DATA:
            for m in range(self.NUM_OF_SUPERVISIONS):
                frame = self.env.reset()
                reward, action, done, info = 0, 0, False, {'ale.lives': 3}

                while (not done):
                    time.sleep(self.SLEEP)
                    self.env.render()
                    action = self.supervision.play(frame, reward, info['ale.lives'])
                    frame, reward, done, info = self.env.step(action)
                    if done:
                        self.supervision.store_match_on_buffer(m)

            self.env.close()
            self.supervision.save_supervision_data()
        else:
            self.supervision.load_supervision_data()
            
        for m in range(self.NUM_OF_SUPERVISIONS):
            num_of_frames = self.supervision.match_buffer[m][0]
            array_of_frames = self.supervision.match_buffer[m][1]
            array_of_actions = self.supervision.match_buffer[m][2]
            array_of_rewards = self.supervision.match_buffer[m][3]
            array_of_lives = self.supervision.match_buffer[m][4]

            for i in range(num_of_frames):
                print(array_of_frames[i].shape)
                print(array_of_rewards[i])
                print(array_of_actions[i])
                print(array_of_lives[i])
                self.simplernn.train(array_of_frames[i], array_of_rewards[i], array_of_lives[i], array_of_actions[i])

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