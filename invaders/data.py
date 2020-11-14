import numpy as np
import random

class NeuralData():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

    buffer_len = 3

    frame_buffer = np.zeros(shape_of_single_frame, dtype=int)
    action_buffer = np.zeros(1, dtype=int)
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2).reshape(self.shape_of_single_frame)

    def store_frame(self, frame):
        frame = self.gray_crop(frame)

        if len(self.frame_buffer) >= self.buffer_len:
            self.frame_buffer = self.frame_buffer[1:-1]
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        else:
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))

    def get_frame_buffer_shape(self):
        return self.frame_buffer.shape

    def get_last_frame(self):
        return self.frame_buffer[-1]

    def store_action(self, action):
        if len(self.action_buffer) >= self.buffer_len:
            self.action_buffer = self.action_buffer[1:-1]
            self.action_buffer = np.append(self.action_buffer, np.array(action))
        else:
            self.action_buffer = np.append(self.action_buffer, np.array(action))

    def get_last_action(self):
        return self.action_buffer[-1]
    
    
class SupervisionData():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

    last_match = 0
    match_buffer = {}

    frame_buffer = np.zeros(shape_of_single_frame, dtype=int)
    action_buffer = np.zeros(1, dtype=int) 
    reward_buffer = np.zeros(1, dtype=int)
    live_buffer = np.zeros(1, dtype=int)
    
    def gray_crop(self, ndarray):
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2).reshape(self.shape_of_single_frame)

    def get_last_action(self):
        return self.action_buffer[-1]

    def is_new_match(self, match):
        return True if ((match - self.last_match) == 1) else False
    
    def store_match(self, frame, reward, match, live, action):
        frame = self.gray_crop(frame)

        self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        self.action_buffer = np.append(self.action_buffer, np.array(action))
        self.reward_buffer = np.append(self.reward_buffer, np.array(reward))
        self.live_buffer = np.append(self.live_buffer, np.array(live))

        if self.is_new_match(match):
            # Apenas para remover dados desnecessarios de inicializacao
            self.frame_buffer = self.frame_buffer[1:-1]
            self.action_buffer = self.action_buffer[1:-1]
            self.reward_buffer = self.reward_buffer[1:-1]
            self.live_buffer = self.live_buffer[1:-1]
            # Apenas para remover dados desnecessarios de inicializacao


            self.match_buffer.update({match : (self.frame_buffer, self.action_buffer, self.reward_buffer, self.live_buffer)})

            
            # Zerar variaveis
            self.frame_buffer = np.zeros(self.shape_of_single_frame, dtype=int)
            self.action_buffer = np.zeros(1, dtype=int)
            self.reward_buffer = np.zeros(1, dtype=int)
            self.live_buffer = np.zeros(1, dtype=int)
            # Zerar variaveis


            self.last_match = match