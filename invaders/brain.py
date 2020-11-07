from data import DataProcessing
import math
import numpy as np
from keras.models import load_model, Sequential
import keras.layers as layers

class Neural():
    """
    docstring
    """

    def __init__(self, **kwargs):
        self.PATH = str(kwargs['path'] + "model")
        self.model = Sequential()
    
    def plot(self):
        pass
    
    def load(self):
        self.model = load_model(self.PATH)
        print("Succesfully loaded network.")

    def save(self):
        self.model.save(self.PATH)
        print("Successfully saved network.")


class SimpleRNN(Neural, DataProcessing):
    """
    docstring
    """
    
    ACTIONS = [0, 1, 2, 3]
    BATCH_SIZE = 10

    def __init__(self, **kwargs):
        self.input_shape = ((self.y_max-self.y_min),(self.x_max-self.x_min))
        self.input_neurons = (self.y_max-self.y_min)*(self.x_max-self.x_min)
        self.output_neurons = len(self.ACTIONS)
        self.hidden_neurons = round(math.sqrt((self.input_neurons*self.output_neurons)))

        self.NUM_OF_TRAINS = kwargs['trains']
        self.VERBOSE = kwargs['verbose']
        self.EPOCHS = kwargs['epochs']

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
    
    def reset_states(self, live):
        if live < self.CURRENT_LIVE or live == 3:
            self.model.reset_states()
    
    def train(self, frame, reward, info):
        self.reset_states(info['ale.lives'])
        
        history = self.model.fit(self.frame_buffer.reshape(self.get_frame_buffer_shape()), self.action_buffer, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        
        self.store_frame(frame)

    def predict(self, frame, reward, info, match):
        self.CURRENT_LIVE = info['ale.lives']

        if match <= self.NUM_OF_TRAINS:
            self.train(frame, reward, info)
            
        self.store_action(self.ACTIONS[np.argmax(self.model.predict_on_batch(self.get_last_frame().reshape(self.shape_of_single_frame)))])

        return self.get_last_action()