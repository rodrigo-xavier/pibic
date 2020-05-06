import time

import keyboard
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

import gym

def ndarray2img(ndarray, colormap):
    import matplotlib.pyplot as plt
    plt.imshow(ndarray, cmap=plt.get_cmap(colormap))
    plt.show()
    # exit()

class DeepLearning():
    ACTIONS = [1, 2, 3]
    model = tf.keras.models.Sequential()
    y_min, y_max, x_min, x_max  = 25, 195, 20, 140
    input_shape = (y_max-y_min, x_max-x_min, 1)

    # def __init__(self):
    #     self.model.add(layers.Flatten())
    #     self.model.add(layers.Dense(250, activation="tanh", bias_initializer="random_uniform"))
    #     self.model.add(layers.Dense(self.ACTIONS, activation="softmax"))
    #     self.model.compile(loss="mean_squared_error",
    #                        optimizer=RMSprop(lr=0.00025,
    #                                          rho=0.95,
    #                                          epsilon=0.01),
    #                        metrics=["accuracy"])

    # def train(self, observation, reward):
    #     data = self.image_processing(observation)
    #     ndarray2img(data, 'gray')

    #     self.model.fit(data, self.ACTIONS,  batch_size=32)
    #     self.model.evaluate()
    #     self.model.predict()

    def image_processing(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)

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