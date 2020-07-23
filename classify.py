import time

import keyboard
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Input
from CircleCollisions2 import main
import matplotlib.pyplot as plt
from utils import show_array_as_img
import cv2
import gym
import math
import os


IMG_SIZE = 256
PNG_PATH = "database/pygame/img/"
NPZ_PATH = "database/pygame/npz/"
PATH = "/home/cyber/GitHub/pibic/pibic/database/pygame"

class PrepareData():
    img_size = IMG_SIZE
    png_path = PNG_PATH
    npz_path = NPZ_PATH + "pygame.npz"
    group_of_images = []

    def get_preprocessed_img(self, img_path):
        import cv2
        img = cv2.imread(img_path, 0) # Convert to grayscale
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32')
        img /= 255       
        return img

    def img2npz(self):
        for f in os.listdir(self.png_path):
            if f.find(".png") != -1:
                img = self.get_preprocessed_img("{}/{}".format(self.png_path, f))
                # show_array_as_img(img, 'gray')
                self.group_of_images.append(img)

        np.savez_compressed(self.npz_path, self.group_of_images)


class DeepLearning():
    model = tf.keras.models.Sequential()
    
    ACTIONS = ["circulo", "quadrado"]

    input_shape = (IMG_SIZE, IMG_SIZE)
    dim_input = (IMG_SIZE)*(IMG_SIZE)
    dim_output = len(ACTIONS)
    nb_units = 5
    # n_hidden_layer = round(math.sqrt((dim_input*dim_output)))

    def __init__(self):
        # self.model.add(
            # layers.Conv2d(
            # )
        # )
        # self.model.add(
        #     layers.Flatten(
        #         input_shape=self.input_shape
        #     )
        # )
        self.model.add(
            layers.SimpleRNN(
                input_shape=(None, self.dim_input), 
                return_sequences=True, 
                units=self.nb_units
            )
        )
        self.model.add(
            layers.TimeDistributed(
                layers.Dense(
                    activation='sigmoid',
                    units=self.dim_output
                )
            )
        )
        self.model.compile(
            loss = 'mse', 
            optimizer = 'rmsprop'
        )
        # self.model.compile(
        #     loss='sparse_categorical_crossentropy',
        #     optimizer='adam',
        #     metrics=['accuracy']
        # )

        self.model.summary()

        from tensorflow.keras.utils.vis_utils import plot_model
        import graphviz
        from interface import implements, Interface
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    def load_data(self):
        pygame = np.load(NPZ_PATH+"pygame.npz")
        self.pygame = pygame.f.arr_0

    def load_another_data(self):
        pygame = np.load("database/pygame/npz2/pygame.npz")
        self.another_pygame = pygame.f.arr_0


    def train(self, epochs, batch_size):
        print(self.model.summary())

        history = self.model.fit(self.pygame, self.ACTIONS, epochs=epochs, batch_size=batch_size)
        self.plot_history(history)

    def predict(self, observation, reward):
        actions = self.model.predict(x=self.another_pygame)
        print(actions)

        selected_action = np.argmax(actions)

        # print(selected_action)
        # print(actions[0, selected_action])

        return selected_action



# main()
# p = PrepareData()
# p.img2npz()


a = DeepLearning()
a.load_data()
a.train(epochs=10, batch_size=32)
a.load_another_data()
a.predict()
