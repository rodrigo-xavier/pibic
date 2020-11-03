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
        self.path = str(kwargs['path'] + "model")
        self.model = Sequential()        
    
    def plot(self):
        pass
    
    def load(self):
        self.model = load_model(self.path)
        print("Succesfully loaded network.")

    def save(self):
        self.model.save(self.path)
        print("Successfully saved network.")


class SimpleRNN(Neural, DataProcessing):
    """
    docstring
    """
    
    ACTIONS = [0, 1, 2, 3]
    RESET_AFTER_MATCHES = 15
    EPOCHS = 3
    BATCH_SIZE = 10
    VERBOSE = True
    NUM_OF_TRAIN = 30

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
    
    def reset_states(self, match):
        if match % self.RESET_AFTER_MATCHES == 0:
            self.model.reset_states()

    def train(self, frame, reward, match):
        # self.reset_states(match)
        self.store_match(frame, reward, match)
        
        if match >= self.matches_len and self.okay:
            history = self.model.fit(self.get_best_match(), self.get_actions_of_best_match(), epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
            self.okay = False

    def predict(self, frame, reward, info, match):
        if match < self.NUM_OF_TRAIN:
            self.train(frame, reward, match)

        self.store_action(self.ACTIONS[np.argmax(self.model.predict_on_batch(self.get_last_frame().reshape(1, 170, 120)))])

        return self.get_last_action()