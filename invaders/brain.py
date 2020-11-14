from data import NeuralData, SupervisionData
import math
import numpy as np
from keras.models import load_model, Sequential
import keras.layers as layers
import keyboard

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


class SimpleRNN(Neural, NeuralData):
    """
    docstring
    """
    
    ACTIONS = [0, 1, 2, 3]
    BATCH_SIZE = 10
    CURRENT_LIVE = 3

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
    
    def reset_states(self, lives):
        if lives < self.CURRENT_LIVE:
            self.model.reset_states()
            self.CURRENT_LIVE = lives
    
    def train(self, frame, reward, lives):
        self.reset_states(lives)
        history = self.model.fit(self.frame_buffer.reshape(self.get_frame_buffer_shape()), self.action_buffer, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        self.store_frame(frame)

    def predict(self, frame, reward, lives):
        self.store_action(self.ACTIONS[np.argmax(self.model.predict_on_batch(self.get_last_frame().reshape(self.shape_of_single_frame)))])
        return self.get_last_action()


class Supervision(SupervisionData):
    ACTION = {
        "NOOP":         0,
        "FIRE":         [" ", 1],
        "RIGHT":        ["j", 2],
        "LEFT":         ["f", 3],
        # "RFIRE":        ["k", 4],
        # "LFIRE":        ["d", 5],
    }

    def __init__(self, **kwargs):
        self.PATH = str(kwargs['path'])
        self.LOAD_SUPERVISION_DATA = kwargs['load_supervision_data']
        self.SAVE_SUPERVISION_DATA_AS_PNG = kwargs['save_supervision_data_as_png']
        self.SAVE_SUPERVISION_DATA_AS_NPZ = kwargs['save_supervision_data_as_npz']
    
    def play(self, frame, reward, live, match):
        self.store_match(frame, reward, match, live, self.supervision_movement())
        return self.get_last_action()
    
    def supervision_movement(self):
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
    
    def save_supervision_data(self):
        if self.SAVE_SUPERVISION_DATA_AS_PNG:
            self.save_as_png()
        if self.SAVE_SUPERVISION_DATA_AS_NPZ:
            self.save_as_npz()
    
    def load_supervision_data(self):
        """
        docstring
        """
        pass

    def save_as_png():
        for m in self.match_buffer:
            pass
        img = Image.fromarray(array)
        img = img.convert("L")

        path = self.PATH + "/img/" + str(counter) + ".png"
        img.save(path)

    def save_as_npz():
        path = self.PATH + "/npz/" + "observation_list.npz"
        np.savez_compressed(path, observation_list)

        path = self.PATH + "/npz/" + "action_list.npz"
        np.savez_compressed(path, action_list)

        path = self.PATH + "/npz/" + "reward_list.npz"
        np.savez_compressed(path, reward_list)

        path = self.PATH + "/npz/" + "live_list.npz"
        np.savez_compressed(path, live_list)

