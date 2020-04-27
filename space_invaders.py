import time

import keyboard
import numpy as np
import tensorflow as tf

import gym

def ndarray2img(ndarray, colormap):
    import matplotlib.pyplot as plt
    plt.imshow(ndarray, cmap = plt.get_cmap(colormap))
    plt.show()
    exit()

class DeepLearning():

    def learning(self, observation, reward):
        gray_scale = self.rgb2gray(observation)
        ndarray2img(gray_scale, 'gray')

    def rgb2gray(self, observation):
        return np.mean(observation, axis=2)

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
    deep = DeepLearning()

    def go(self, match, run_choice):
        for m in range(match):
            if run_choice == 1:
                self.run_automatic()
            elif run_choice == 2:
                self.run_manual()
            elif run_choice == 3:
                self.run_deep_learning()
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
    
    def run_deep_learning(self):
        observation = self.env.reset()
        reward = 0
        done = False
        while (not done):
            self.env.render()
            action = self.deep.learning(observation, reward)
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