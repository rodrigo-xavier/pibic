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
    RESET_STATES = 5
    EPOCHS = 10
    BATCH_SIZE = 10
    count_states = 0
    list_of_probabilities = []
    total_reward = 0

    last_frame = []
    penultimate_frame = []
    last_action = 0
    penultimate_action = 0

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

    def get_mov_reward(self, reward):
        diff = self.total_reward - reward
        self.total_reward = reward

        return diff
    
    def reset_states(self):
        self.count_states += 1
        if self.count_states >= self.RESET_STATES:
            self.model.reset_states()
            self.count_states = 0
    
    def predict(self, frame, reward):
        reward = self.get_mov_reward(reward)

        if reward > 0 and self.full_buffer:
            self.reset_states()
            history = self.model.fit(self.frame_buffer.reshape(self.get_shape()), self.action_buffer, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)

        self.store_frame(frame)
        self.store_action(self.ACTIONS[np.argmax(self.model.predict_on_batch(self.get_last_frame().reshape(1, 170, 120)))])

        return self.get_last_action()




        
        
