import time

import keyboard
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import show_array_as_img, store_csv, store_png, store_npz, arguments
import matplotlib.pyplot as plt

import gym
import math


PATH = "/home/cyber/GitHub/pibic/pibic/database"

class DeepLearning():
    model = tf.keras.models.Sequential()
    
    ACTIONS = [0, 1, 2, 3]
    y_min, y_max, x_min, x_max  = 25, 195, 20, 140
    input_shape = (y_max-y_min, x_max-x_min, 1)

    # Quantidade de Neuronios em cada camada
    n_input_layer = (y_max-y_min)*(x_max-x_min+1) # +1 Para considerar a entrada de actions
    n_output_layer = len(ACTIONS)
    n_hidden_layer = round(math.sqrt((n_input_layer*n_output_layer)))

    def __init__(self):
        self.model.add(
            layers.Flatten(
                input_shape=self.input_shape
            )
        )
        self.model.add(
            layers.Dense(
                units=self.n_hidden_layer,
                activation='tanh',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'
            )
        )
        self.model.add(
            layers.Dense(
                units=self.n_output_layer,
                activation='softmax'
            )
        )
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print(self.model.summary())
        self.load_data()
        self.train()
    
    def load_data(self):
        self.observation_list = np.load(PATH + '/npz/observation_list.npz')
        self.action_list = np.load(PATH + '/npz/action_list.npz')

    def train(self):
        self.model.fit(self.observation_list, self.action_list, epochs=150, batch_size=10)
        self.plot_history()

    def predict(self, observation):
        processed_observation = gray_crop(observation)

        action = self.model.predict(x=processed_observation)
        print(action)
        # self.model.evaluate()
        return action

    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)

    def plot_history(self):
        # Plot training & validation accuracy values
        plt.plot(self.model.history['acc'])
        plt.plot(self.model.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.model.history['loss'])
        plt.plot(self.model.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


    # def train(self, observation, reward):
    #     data = self.gray_crop(observation)
    #     # ndarray2img(data, 'gray')

    #     self.history = self.model.fit(
    #                         x=data,
    #                         y=self.ACTIONS,
    #                         batch_size=32,
    #                         epochs=50,
    #                         verbose=1,
    #                     )

    #     self.model.evaluate()
    #     self.model.predict()
    
    # def store_model_graph(self):
    #     from keras.utils import plot_model
    #     plot_model(self.model, to_file='model.png')

class PlayLearning():
    # ACTION = {
    #     "NOOP":         0,
    #     "FIRE":         [" ", 1],
    #     "RIGHT":        ["j", 2],
    #     "LEFT":         ["f", 3],
    #     "RFIRE":        ["k", 4],
    #     "LFIRE":        ["d", 5],
    # }

    env = gym.make("SpaceInvaders-v0")
    # deeplearning = DeepLearning()

    def run(self, match):
        for m in range(match):
            self.play()
        self.env.close()
    
    def play(self):
        observation = self.env.reset()
        reward = 0
        done = False
        
        while (not done):
            self.env.render()
            action = self.deeplearning.predict(observation, reward)
            observation, reward, done, info = self.env.step(action)
            time.sleep(0.2)


class PlayManual():
    ACTION = {
        "NOOP":         0,
        "FIRE":         [" ", 1],
        "RIGHT":        ["j", 2],
        "LEFT":         ["f", 3],
        "RFIRE":        ["k", 4],
        "LFIRE":        ["d", 5],
    }
    y_min, y_max, x_min, x_max  = 25, 195, 20, 140
    observation_list = []
    action_list = []
    # y = 175, x = 120 | x*y = 21000

    env = gym.make("SpaceInvaders-v0")

    def run(self, match):
        for m in range(match):
            self.play()
        self.env.close()

    def run_storing(self, match):
        for m in range(match):
            self.play_and_store()
        self.env.close()

    def play(self):
        observation = self.env.reset()
        done = False

        while (not done):
            self.env.render()
            action = self.controller()
            observation, reward, done, info = self.env.step(action)
            time.sleep(0.1)

    def play_and_store(self):
        observation = self.env.reset()
        done = False
        counter = 0

        # y = 175, x = 121 | x*y = 21175 bytes
        while (not done):
            counter+=1
            self.env.render()
            action = self.controller()
            observation, reward, done, info = self.env.step(action)
            time.sleep(0.05)

            processed_observation = self.gray_crop(observation)

            store_png(action, processed_observation, counter)
            store_csv(action, processed_observation, counter)

            self.observation_list.append(processed_observation)
            self.action_list.append(action)

        store_npz(self.observation_list, self.action_list)

    def controller(self):
        if keyboard.is_pressed(self.ACTION["RIGHT"][0]):
            return self.ACTION["RIGHT"][1]
        elif keyboard.is_pressed(self.ACTION["LEFT"][0]):
            return self.ACTION["LEFT"][1]
        elif keyboard.is_pressed(self.ACTION["FIRE"][0]):
            return self.ACTION["FIRE"][1]
        elif keyboard.is_pressed(self.ACTION["RFIRE"][0]):
            return self.ACTION["RFIRE"][1]
        elif keyboard.is_pressed(self.ACTION["LFIRE"][0]):
            return self.ACTION["LFIRE"][1]
        else:
            return self.ACTION["NOOP"]
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)


class PlayAutomatic():
    env = gym.make("SpaceInvaders-v0")

    def run(self, match):
        for m in range(match):
            self.play()
        self.env.close()

    def play(self):
        observation = self.env.reset()
        done = False

        while (not done):
            self.env.render()
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            # ndarray_to_img(observation)