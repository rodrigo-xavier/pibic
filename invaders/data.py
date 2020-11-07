import numpy as np
import random

class DataProcessing():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

    buffer_len = 3
    absolute_reward = 0

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
    
    def get_absolute_reward(self, reward):
        self.absolute_reward = reward + self.absolute_reward