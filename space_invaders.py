import time

import keyboard
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import ndarray2img
import matplotlib.pyplot as plt

import gym


class DeepLearning():
    ACTIONS = [1, 2, 3] 
    model = tf.keras.models.Sequential()
    y_min, y_max, x_min, x_max  = 25, 195, 20, 140
    input_shape = (y_max-y_min, x_max-x_min, 1)

    def __init__(self):
        self.model.add(
            layers.Dense(
                units=174, # ((y_max-y_min)*(x_max-x_min)*255)^(1/3) = 174
                input_shape=self.input_shape,
                kernel_initializer='random_uniform', 
                bias_initializer='zeros'
            )
        )
        self.model.add(layers.Flatten())
        self.model.add(
            layers.Dense(
                units=174,
                activation='tanh'
            )
        )
        self.model.add(
            layers.Dense(
                units=3,
                activation='softmax'
            )
        )
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, observation, reward):
        data = self.image_processing(observation)
        # ndarray2img(data, 'gray')

        self.history = self.model.fit(
                            x=data, 
                            y=self.ACTIONS, 
                            batch_size=32, 
                            epochs=50,
                            verbose=1,
                        )
        plot_history()

        # return self.history.history[]

        # self.model.evaluate()
        # self.model.predict()

    def image_processing(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)
    
    def plot_history(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    def store_model_graph(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')

class Play():
    ACTION = {
        "NOOP":         0,
        "FIRE":         [" ", 1],
        "RIGHT":        ["j", 2],
        "LEFT":         ["f", 3],
        "RFIRE":        ["k", 4],
        "LFIRE":        ["d", 5],
    }

    env = gym.make('SpaceInvaders-v0')
    deeplearning = DeepLearning()

    def go(self, match, run_choice):
        for m in range(match):
            if run_choice == 1:
                self.run_automatic()
            elif run_choice == 2:
                self.run_manual()
            elif run_choice == 3:
                self.run_neural_network()
        self.env.close()

    def run_automatic(self):
        observation = self.env.reset()
        done = False
        while (not done):
            self.env.render()
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            # ndarray_to_img(observation)

    def run_manual(self):
        observation = self.env.reset()
        done = False
        while (not done):
            self.env.render()
            action = self.controller()
            observation, reward, done, info = self.env.step(action)
            time.sleep(0.1)
    
    def run_neural_network(self):
        observation = self.env.reset()
        reward = 0
        done = False
        while (not done):
            self.env.render()
            self.deeplearning.train(observation, reward)
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)

    def controller(self):
        if keyboard.is_pressed(ACTION["RIGHT"][0]):
            return ACTION["RIGHT"][1]
        elif keyboard.is_pressed(ACTION["LEFT"][0]):
            return ACTION["LEFT"][1]
        elif keyboard.is_pressed(ACTION["FIRE"][0]):
            return ACTION["FIRE"][1]
        elif keyboard.is_pressed(ACTION["RFIRE"][0]):
            return ACTION["RFIRE"][1]
        elif keyboard.is_pressed(ACTION["LFIRE"][0]):
            return ACTION["LFIRE"][1]
        else:
            return ACTION["NOOP"]