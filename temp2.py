import gym
import time
import keyboard
from PIL import Image
# import tensorflow as tf
# import numpy as np


ACTION = {
    "NOOP":         0,
    "FIRE":         [" ", 1],
    "RIGHT":        ["j", 2],
    "LEFT":         ["f", 3],
    "RFIRE":        ["k", 4],
    "LFIRE":        ["d", 5],
}


class Play():
    env = gym.make('SpaceInvaders-v0')

    def go(self, match, run_mode):
        for m in range(match):
            if run_mode == "Automatic":
                self.automatic()
            else:
                self.manual()
        self.close()
    
    def close(self):
        self.env.close()

    def automatic(self):
        observation = self.env.reset()
        done = False
        while (not done):
            self.env.render()
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            # self.print_image(observation)
            # time.sleep(0.1)

    def manual(self):
        observation = self.env.reset()
        done = False
        while (not done):
            self.env.render()
            action = self.controller()
            observation, reward, done, info = self.env.step(action)
            time.sleep(0.1)

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


    def print_image(self, observation):
        # print(self.env.action_space)
        # print(self.env.observation_space)
        # print(self.env.observation_space.high)
        # print(self.env.observation_space.low)         
        w, h = len(observation), len(observation[0])
        data = observation
        img = Image.fromarray(data, 'RGB')
        img.save('created_screen.png')
        img.show()


class DeepLearning():
    pass