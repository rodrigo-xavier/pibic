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
    RESET_STATES = 10
    EPOCHS = 1
    BATCH_SIZE = 10
    VERBOSE = True
    NUM_OF_TRAIN = 50
    last_reward = 0
    last_info = 0
    saved = False

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
        diff = 0

        if reward != 0:
            diff = reward - self.last_reward
            self.last_reward = reward

        return diff
    
    def reset_states(self, match):
        if match % self.RESET_STATES == 0:
            self.model.reset_states()
    
    def train_frames(self, frame, reward, info):
        reward = self.get_mov_reward(reward)

        if reward > 0 and self.full_buffer:
            self.reset_states()
            history = self.model.fit(self.frame_buffer.reshape(self.get_shape()), self.action_buffer, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        elif info['ale.lives'] < self.last_info and self.full_buffer:
            history = self.model.fit(self.frame_buffer.reshape(self.get_shape()), self.get_evasive_action(), epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)

        self.store_frame(frame)

    def predict_frames(self, frame, reward, info, match):
        if match < self.NUM_OF_TRAIN:
            self.train_frames(frame, reward, info)
        self.store_action(self.ACTIONS[np.argmax(self.model.predict_on_batch(self.get_last_frame().reshape(1, 170, 120)))])
        self.last_info = info['ale.lives']

        return self.get_last_action()

    def train_matches(self, frame, reward, match):
        # self.reset_states(match)
        self.store_match(frame, reward, match)
        
        if (match % self.matches_len == 0) and self.okay:
            history = self.model.fit(self.get_best_match(), self.get_actions_of_best_match(), epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
            self.okay = False

    def predict_matches(self, frame, reward, info, match):
        if match < self.NUM_OF_TRAIN:
            self.train_matches(frame, reward, match)
        elif match == self.NUM_OF_TRAIN and not(self.saved):
            self.save()
            self.saved = True

        self.store_action(self.ACTIONS[np.argmax(self.model.predict_on_batch(self.get_last_frame().reshape(1, 170, 120)))])

        return self.get_last_action()
