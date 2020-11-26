from data import NeuralData
import math
import numpy as np
from keras.models import load_model, Sequential
import keras.layers as layers
import keyboard
import os

class Neural():
    """
    docstring
    """

    def __init__(self, **kwargs):
        self.PATH = str(kwargs['path'] + "model/")
        self.model = Sequential()
        self.hidden_neurons = kwargs['hidden_neurons']
    
    def plot(self):
        pass
    
    def load(self):
        self.model = load_model(self.PATH + self.next_folder)
        print("Succesfully loaded network.")

    def save(self):
        m = []

        for folder in os.listdir(self.PATH):
            m.append(int(folder.split("_")[1]))
        if m:
            number_of_last_folder = max(m)
        else:
            number_of_last_folder = 0
        self.next_folder = "net_" + str(number_of_last_folder + 1) + "_epochs=" + str(self.EPOCHS) + "_hidden=" + str(self.hidden_neurons) + "_reset=False"

        os.mkdir(self.PATH + self.next_folder)

        self.model.save(self.PATH + self.next_folder)
        print("Successfully saved network.")


class SimpleRNN(Neural, NeuralData):
    """
    docstring
    """
    
    ACTIONS = [0, 1, 2, 3]
    BATCH_SIZE = 1
    last_life = 3
    last_match = 0
    reset_states_count = 0
    shape_of_single_action = (1, len(ACTIONS))

    def __init__(self, **kwargs):
        self.input_shape = ((self.y_max-self.y_min),(self.x_max-self.x_min))
        self.input_neurons = (self.y_max-self.y_min)*(self.x_max-self.x_min)
        self.output_neurons = len(self.ACTIONS)

        if 'hidden_neurons' in kwargs:
            self.hidden_neurons = kwargs['hidden_neurons']
        else:
            self.hidden_neurons = round(math.sqrt((self.input_neurons*self.output_neurons)))

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

    def prepare_action_data(self, action):
        new_action = np.zeros((len(action), len(self.ACTIONS)), dtype=int) 

        for i in range(len(action)):
            new_action[i, self.ACTIONS.index(action[i])] = 1
    
        return new_action
    
    def was_killed(self, life):
        return True if ((self.last_life - life) == 1) else False
    
    def is_new_match(self, match):
        return True if ((match - self.last_match) == 1) else False
    
    def reset_states(self, life, match):
        if self.was_killed(life) or self.is_new_match(match):
            self.model.reset_states()
            self.last_life = life
            self.last_match = match

            print("reseted states")
            self.reset_states_count += 1
    
    def train(self, frame, life, action, match, num_of_frames):
        frames = frame.reshape((num_of_frames, 170, 120))
        action = self.prepare_action_data(action)

        # for j in range(self.EPOCHS):
        #     for i in range(num_of_frames):
        #         history = self.model.fit(
        #             frames[i].reshape(self.shape_of_single_frame), 
        #             action[i].reshape(self.shape_of_single_action), 
        #             epochs=1, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE
        #         )
                # print(self.model.get_weights())
                # self.reset_states(life[i], match)
            # print(j)

        history = self.model.fit(frames, action, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)

    def predict(self, frame):
        return self.ACTIONS[np.argmax(self.model.predict(self.gray_crop(frame)))]


class Reinforcement(NeuralData):
    ACTION = {
        "NOOP":         0,
        "FIRE":         [" ", 1],
        "RIGHT":        ["j", 2],
        "LEFT":         ["f", 3],
        # "RFIRE":        ["k", 4],
        # "LFIRE":        ["d", 5],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def play(self, frame, reward, live):
        self.store_data_on_buffer(frame, reward, live, self.reinforcement_movement())
        return self.get_last_action()
    
    def reinforcement_movement(self):
        if keyboard.is_pressed(self.ACTION["RIGHT"][0]):
            return self.ACTION["RIGHT"][1]
        elif keyboard.is_pressed(self.ACTION["LEFT"][0]):
            return self.ACTION["LEFT"][1]
        elif keyboard.is_pressed(self.ACTION["FIRE"][0]):
            return self.ACTION["FIRE"][1]
        # elif keyboard.is_pressed(self.ACTION["RFIRE"][0]):
        #     return self.ACTION["RFIRE"][1]
        # elif keyboard.is_pressed(self.ACTION["LFIRE"][0]):
        #     return self.ACTION["LFIRE"][1]
        else:
            return self.ACTION["NOOP"]