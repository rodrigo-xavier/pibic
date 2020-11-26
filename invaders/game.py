import gym
from brain import SimpleRNN, Reinforcement

class Invaders():
    """
    docstring
    """

    env = gym.make("SpaceInvaders-v0")

    def __init__(self, **kwargs):
        self.PATH = str(kwargs['path'])
        self.MATCHES = kwargs['matches']
        self.SLEEP = kwargs['sleep']
        
        self.simplernn = SimpleRNN(**kwargs)
        self.reinforcement = Reinforcement(path=self.PATH)
    
    def save_reinforcement(self):
        import time
        while input("Do you want to play (y/n)? ")=='y':
            self.reinforcement.reset()
            frame = self.env.reset()
            reward, action, done, info = 0, 0, False, {'ale.lives': 3}

            while (not done):
                time.sleep(self.SLEEP)
                self.env.render()
                action = self.reinforcement.play(frame, reward, info['ale.lives'])
                frame, reward, done, info = self.env.step(action)
                
            self.reinforcement.save_data()
        del self.reinforcement

    def load_reinforcement_and_train(self):
        matches = self.reinforcement.num_of_samples()
        
        for m in range(matches):
            try:
                num_of_frames, frames, actions, rewards, lifes = self.reinforcement.load_npz(m)
                self.simplernn.train(frames, lifes, actions, m, num_of_frames)
            except:
                print("Training")
                print("Can't load folder match_" + str(m))

        self.simplernn.save()
    
    def test_overfitting(self):
        import numpy as np

        self.simplernn.load()
        matches = self.reinforcement.num_of_samples()
        success, accuracy, accumulated_loss, loss, fail = 0, 0, 0, 0, 0
        
        for m in range(matches):
            try:
                num_of_frames, frames, actions, rewards, lifes = self.reinforcement.load_npz(m)

                for i in range(num_of_frames):
                    predicted = self.simplernn.model.predict(frames[i].reshape(self.simplernn.shape_of_single_frame))
                    result = self.simplernn.ACTIONS[np.argmax(predicted)]
                    
                    if result==actions[i]:
                        success += 1
                        accuracy = (success*100)/num_of_frames
                    else:
                        fail += 1
                        accumulated_loss = accumulated_loss + (result - actions[i])**2
                        loss = accumulated_loss/fail
                    
                print("Accuracy: " + str(accuracy) + "% Loss: " + str(loss) + "%")
            except:
                print("Overfitting")
                print("Can't load folder match_" + str(m))

        del self.reinforcement
        
    def run_predict(self):
        for m in range(self.MATCHES):
            frame = self.env.reset()
            reward, action, done, info = 0, 0, False, {'ale.lives': 3}

            while (not done):
                self.env.render()
                action = self.simplernn.predict(frame)
                frame, reward, done, info = self.env.step(action)
            
        self.env.close()


# class AutomaticTest():
#     """
#     docstring
#     """

#     env = gym.make("SpaceInvaders-v0")

#     def __init__(self, **kwargs):
#         self.MATCHES = kwargs['matches']
#         self.SLEEP = kwargs['sleep']
        
#         self.simplernn = SimpleRNN(**kwargs)
#         self.reinforcement = Reinforcement(**kwargs)

#     def load_reinforcement_and_train(self):
#         self.reinforcement.load_npz()
        
#         for m in range(self.NUM_OF_REINFORCEMENTS):
#             num_of_frames = self.reinforcement.match_buffer[m][0]
#             frames = self.reinforcement.match_buffer[m][1]
#             actions = self.reinforcement.match_buffer[m][2]
#             rewards = self.reinforcement.match_buffer[m][3]
#             lifes = self.reinforcement.match_buffer[m][4]

#             self.simplernn.train(frames, rewards, lifes, actions, m, num_of_frames)

#         self.simplernn.save()
#         del self.reinforcement
    
#     def test_overfitting(self):
#         import numpy as np
#         self.reinforcement.load_npz()
#         self.simplernn.load()
#         success = 0
#         fail = 0
        
#         for m in range(self.NUM_OF_REINFORCEMENTS):
#             num_of_frames = self.reinforcement.match_buffer[m][0]
#             frames = self.reinforcement.match_buffer[m][1]
#             actions = self.reinforcement.match_buffer[m][2]

#             for i in range(num_of_frames):
#                 result = self.simplernn.ACTIONS[np.argmax(self.simplernn.model.predict(frames[i].reshape(1, 170, 120)))]
                
#                 if result==actions[i]:
#                     success += 1
#                 else:
#                     fail += 1
                
#                 print("Success: " + str((success*100)/num_of_frames) + "% Fail: " + str((fail*100)/num_of_frames) + "%")

#         del self.reinforcement