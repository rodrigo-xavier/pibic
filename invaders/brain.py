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
        self.LOAD_MODEL = kwargs['load_model']
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
    BATCH_SIZE = 32
    last_life = 3
    last_match = 0
    reset_states_count = 0

    total_frames = 16521
    frames_until_now = 0

    def __init__(self, **kwargs):
        self.input_shape = ((self.y_max-self.y_min),(self.x_max-self.x_min))
        self.input_neurons = (self.y_max-self.y_min)*(self.x_max-self.x_min)
        self.output_neurons = len(self.ACTIONS)
        self.hidden_neurons = round(math.sqrt((self.input_neurons*self.output_neurons)))

        self.SUPERVISION = kwargs['supervision']
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
    
    def train(self, frame, reward, life, action, match):
        self.reset_states(life, match)

        if not self.SUPERVISION:
            self.store_frame_on_buffer(self.gray_crop(frame))
        else:
            self.store_frame_on_buffer(frame.reshape(self.shape_of_single_frame))

        self.store_action_on_buffer(action)

        if len(self.frame_buffer) % self.buffer_len == 0:
            history = self.model.fit(self.frame_buffer, self.action_buffer, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
            
            # eh possivel prever quanto tempo vai durar o treino tambem
            self.frames_until_now = self.frames_until_now + len(self.frame_buffer)
            print(str((self.frames_until_now*100)/self.total_frames) + " %")

            self.reset_buffer()
            

    def predict(self, frame):
        frame = self.gray_crop(frame)
        # return self.ACTIONS[np.argmax(self.model.predict_on_batch(frame.reshape(self.shape_of_single_frame)))]
        return self.ACTIONS[np.argmax(self.model.predict(frame.reshape(self.shape_of_single_frame), use_multiprocessing=True))]


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
        super().__init__(**kwargs)
    
    def play(self, frame, reward, live):
        self.store_data_on_buffer(frame, reward, live, self.supervision_movement())
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