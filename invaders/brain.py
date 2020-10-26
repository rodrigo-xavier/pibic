from data import DataPreProcessing
import math

class Neural():
    """
    docstring
    """

    def __init__(self, **kwargs):
        self.input_neurons = (kwargs['y_max']-kwargs['y_min'])*(kwargs['x_max']-kwargs['x_min'])
        self.output_neurons = len(kwargs['ACTIONS'])
        self.hidden_neurons = round(math.sqrt((self.input_neurons*self.output_neurons)))
    
    

class LSTM(Neural, DataPreProcessing):
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    
    ACTIONS = [0, 1, 2, 3]

    def __init__(self, **kwargs):
        kwargs.update({'y_min':self.y_min, 'y_max':self.y_max, 'x_min':self.x_min, 'x_max':self.x_max})
        super().__init__(**kwargs)
    
    def train(self, observation, reward):
        observation = self.gray_crop(observation)
