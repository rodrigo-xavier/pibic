import numpy as np
import random

class DataProcessing():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    buffer_shape = (1, (y_max-y_min),(x_max-x_min))
    matches_len = 2
    match_buffer = {}
    frame_buffer = np.zeros(buffer_shape, dtype=int)
    action_buffer = np.zeros(1, dtype=int)
    last_match = 0
    reward_of_best_match = 0.0
    best_match = ()
    absolute_reward = 0
    okay = False
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2).reshape(self.buffer_shape)

    def store_action(self, action):
        if random.random() < 0.025:
            self.action_buffer = np.append(self.action_buffer, np.array(2))
        # elif random.random() < 0.10:
            # self.action_buffer = np.append(self.action_buffer, np.array(3))
        # elif random.random() < 0.25:
        #     self.action_buffer = np.append(self.action_buffer, np.array(1))
        else:
            self.action_buffer = np.append(self.action_buffer, np.array(action))
        
    def get_last_frame(self):
        return self.frame_buffer[-1]

    def get_last_action(self):
        return self.action_buffer[-1]
    
    def is_new_match(self, match):
        return True if ((match - self.last_match) == 1) else False
    
    def get_absolute_reward(self, reward):
        self.absolute_reward = reward + self.absolute_reward
    
    def store_match(self, frame, reward, match):
        frame = self.gray_crop(frame)
        self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        self.get_absolute_reward(reward)
        
        if self.is_new_match(match):
            self.match_buffer.update({(match % self.matches_len) : (self.frame_buffer, self.action_buffer)})

            if self.absolute_reward >= self.reward_of_best_match:
                self.reward_of_best_match = self.absolute_reward
                print(self.reward_of_best_match)
                self.best_match = self.match_buffer[match % self.matches_len]
            
            self.frame_buffer = np.zeros(self.buffer_shape, dtype=int)
            self.action_buffer = np.zeros(1, dtype=int)

            self.last_match = match
            self.absolute_reward = 0
            self.okay = True
    
    def get_shape_of_best_match(self):
            return self.best_match[0].shape
    
    def get_best_match(self):
            return self.best_match[0].reshape(self.get_shape_of_best_match())

    def get_actions_of_best_match(self):
            return self.best_match[1]