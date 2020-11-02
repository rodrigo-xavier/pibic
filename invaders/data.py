import numpy as np
import random

class DataProcessing():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    buffer_shape = (1, (y_max-y_min),(x_max-x_min))
    buffer_len = 999999
    matches_len = 2
    match_buffer = {}
    frame_buffer = np.zeros(buffer_shape, dtype=int)
    action_buffer = np.zeros(1, dtype=int)
    full_buffer = False
    last_match = 0
    best_reward = 0
    reward_of_best_match = 0
    best_match = 0
    absolute_reward = 0
    okay = False
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2).reshape(self.buffer_shape)
    
    def store_frame(self, frame):
        frame = self.gray_crop(frame)

        if len(self.frame_buffer) >= self.buffer_len:
            self.full_buffer = True
            self.frame_buffer = self.frame_buffer[1:-1]
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        else:
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        
    def get_last_frame(self):
        return self.frame_buffer[-1]
    
    def get_shape(self):
        return self.frame_buffer.shape

    def store_action(self, action):
        if len(self.action_buffer) >= self.buffer_len:
            self.action_buffer = self.action_buffer[1:-1]
            self.action_buffer = np.append(self.action_buffer, np.array(action))
        else:
            if random.random() < 0.1:
                self.action_buffer = np.append(self.action_buffer, np.array(random.randint(2, 3)))
            else:
                self.action_buffer = np.append(self.action_buffer, np.array(action))
        
    def get_last_action(self):
        return self.action_buffer[-1]
    
    def get_evasive_action(self):
        action = np.zeros(self.action_buffer.shape, dtype=int)
        action[:] = 2
        return action

    def get_aleatory_action(self):
        action = np.zeros(self.action_buffer.shape, dtype=int)
        action[:] = random.randint(0,3)
        return action
    
    def is_new_match(self, match):
        return True if ((match - self.last_match) == 1) else False
    
    def store_absolute_reward(self, reward):
        if reward > 0:
            self.absolute_reward = reward
    
    def store_match(self, frame, reward, match):
        if match % self.matches_len == 0:
            self.match_buffer = {}
        
        self.store_absolute_reward(reward)

        frame = self.gray_crop(frame)

        self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        
        if self.is_new_match(match):
            self.match_buffer.update({(match % self.matches_len) : (self.frame_buffer, self.action_buffer)})

            print(self.absolute_reward)

            if self.absolute_reward >= self.best_reward:
                self.best_reward = self.absolute_reward
                self.best_match = match % self.matches_len
                self.store_best_match(self.match_buffer[self.best_match], self.best_reward)
            
            self.frame_buffer = np.zeros(self.buffer_shape, dtype=int)
            self.action_buffer = np.zeros(1, dtype=int)

            self.last_match = match
            self.okay = True
    
    def store_best_match(self, match, reward):
        self.best_match = match
        self.reward_of_best_match = reward
    
    def get_shape_of_best_match(self):
        if self.best_reward > self.reward_of_best_match:
            return self.match_buffer[self.best_match][0].shape
        else:
            return self.best_match[0].shape
    
    def get_best_match(self):
        if self.best_reward > self.reward_of_best_match:
            print(self.best_reward)
            print('best reward')
            return self.match_buffer[self.best_match][0].reshape(self.get_shape_of_best_match())
        else:
            print(self.reward_of_best_match)
            print('reward_of_best_match')
            return self.best_match[0].reshape(self.get_shape_of_best_match())

    def get_actions_of_best_match(self):
        if self.best_reward > self.reward_of_best_match:
            return self.match_buffer[self.best_match][1]
        else:
            return self.best_match[1]