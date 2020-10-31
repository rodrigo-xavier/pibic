import numpy as np

class DataProcessing():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    buffer_len = 10
    frame_buffer = [[]]
    action_buffer = []
    full_buffer = False

    def __init__(self, **kwargs):
        self.y_min = kwargs['y_min']
        self.y_max = kwargs['y_max']
        self.x_min = kwargs['x_min']
        self.x_max = kwargs['x_max']
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)
    
    def store_frame(self, frame):
        frame = self.gray_crop(frame)

        if len(self.frame_buffer) <= self.buffer_len:
            print(frame)
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))
            print(self.frame_buffer)
        else:
            self.full_buffer = True
            self.frame_buffer = self.frame_buffer[1:-1]
        
    def get_last_frame(self):
        return self.frame_buffer[-1]
    
    def get_shape(self):
        shape = list(self.frame_buffer.shape)
        return tuple(shape.insert(0, self.frame_buffer.size))

    def store_action(self, action):
        if len(self.action_buffer) <= self.buffer_len:
            self.action_buffer = np.concatenate((self.action_buffer, np.array(action)))
        else:
            self.action_buffer = self.action_buffer[1:-1]
        
    def get_last_action(self):
        return self.action_buffer[-1]