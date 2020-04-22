import random
import time
from collections import Counter, deque
from datetime import datetime

import keyboard
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, Flatten
# , flatten, fully_connected

import gym


class DeepLearning():
    stack_size = 4 # We stack 4 composite frames in total
    # Initialize deque with zero-images one array for each image. Deque is a special kind of queue that deletes last entry when new entry comes in
    stacked_frames  =  deque([np.zeros((88,80), dtype=np.int) for i in range(stack_size)], maxlen=4)

    tf.compat.v1.reset_default_graph()
    #Reset is technically not necessary if variables done  in TF2


    def preprocess_observation(self, observation):
        img = observation[25:201:2, ::2]    # Crop and resize the image
        img = img.mean(axis=2)              # Convert the image to greyscale
        img[img==color] = 0                 # Improv    e image contrast
        img = (img - 128) / 128 - 1         # Next we normalize the image from -1 to +1
        return img.reshape(88,80)

    def stack_frames(self, stacked_frames, state, is_new_episode):
        frame = self.preprocess_observation(state)
        
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((88,80), dtype=np.int) for i in range(stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x, apply elementwise maxima
            maxframe = np.maximum(frame,frame)
            stacked_frames.append(maxframe)
            stacked_frames.append(maxframe)
            stacked_frames.append(maxframe)
            stacked_frames.append(maxframe)
            
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)
            
        else:
            #Since deque append adds t right, we can fetch rightmost element
            maxframe=np.maximum(stacked_frames[-1],frame)
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(maxframe)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2) 
        
        return stacked_state, stacked_frames

    def q_network(self, X, name_scope):
        
        # Initialize layers
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)

        with tf.compat.v1.variable_scope(name_scope) as scope: 


            # initialize the convolutional layers
            layer_1 = conv2d(X, num_outputs=32, kernel_size=(8,8), stride=4, padding='SAME', weights_initializer=initializer) 
            tf.compat.v1.summary.histogram('layer_1',layer_1)
            
            layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4,4), stride=2, padding='SAME', weights_initializer=initializer)
            tf.compat.v1.summary.histogram('layer_2',layer_2)
            
            layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3,3), stride=1, padding='SAME', weights_initializer=initializer)
            tf.compat.v1.summary.histogram('layer_3',layer_3)
            
            # Flatten the result of layer_3 before feeding to the fully connected layer
            flat = flatten(layer_3)
            # Insert fully connected layer
            fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
            tf.compat.v1.summary.histogram('fc',fc)
            #Add final output layer
            output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
            tf.compat.v1.summary.histogram('output',output)
            

            # Vars will store the parameters of the network such as weights
            vars = {v.name[len(scope.name):]: v for v in tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)} 
            #Return both variables and outputs together
            return vars, output

    def epsilon_greedy(self, action, step):
        p = np.random.random(1).squeeze() #1D entries returned using squeeze
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps) #Decaying policy with more steps
        if p< epsilon:
            return np.random.randint(n_outputs)
        else:
            return action

            
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
            # self.rebuild_observation(observation)

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
        done = False
        while (not done):
            deep.preprocess_observation(observation)
            self.env.render()
            action = 1
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

    def rebuild_observation(self, observation):
        w, h = len(observation), len(observation[0])
        data = observation
        img = Image.fromarray(data, 'RGB')
        img.save('created_screen.png')
        img.show()
        exit()


