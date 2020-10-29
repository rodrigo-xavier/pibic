from data import DataPreProcessing
import math
import numpy as np
from keras.models import load_model, Sequential
import keras.layers as layers

class Neural():
    """
    docstring
    """

    def __init__(self, **kwargs):
        self.path = str(kwargs['path'] + "model")
        self.model = Sequential()        
    
    def predict(self, data):
        return self.model.predict(data, batch_size = 1)
    
    def plot(self):
        pass
    
    def load(self):
        self.model = load_model(self.path)
        print("Succesfully loaded network.")

    def save(self):
        self.model.save(self.path)
        print("Successfully saved network.")


class LSTM(Neural, DataPreProcessing):
    """
    docstring
    """
    
    ACTIONS = [0, 1, 2, 3]

    def __init__(self, **kwargs):
        self.input_shape = ((self.y_max-self.y_min),(self.x_max-self.x_min))
        self.input_neurons = (self.y_max-self.y_min)*(self.x_max-self.x_min)
        self.output_neurons = len(self.ACTIONS)
        self.hidden_neurons = round(math.sqrt((self.input_neurons*self.output_neurons)))

        super().__init__(**kwargs)

        self.build()
    
    def build(self):
        self.model.add(
            layers.SimpleRNN(
                units=self.hidden_neurons,
                input_shape=self.input_shape,
                activation='tanh',
                kernel_initializer='random_uniform',
            )
        )
        self.model.add(
            layers.Dense(
                self.output_neurons,
                activation='sigmoid'
            )
        )
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        print("Successfully constructed networks.")
    
    def train(self, observation, reward, action):
        observation = self.gray_crop(observation)

        target = self.model.predict(observation, batch_size = 1)
        targets[i, action[i]] = reward[i]

        loss = self.model.train_on_batch(observation, targets)
        print("We had a loss equal to ", loss)

        return self.predict(observation)

        
        
