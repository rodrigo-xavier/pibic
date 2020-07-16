import time

import keyboard
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import show_array_as_img, store_csv, store_png, store_npz, arguments
import matplotlib.pyplot as plt

import gym
import math


PATH = "/home/cyber/GitHub/pibic/pibic/database/pygame/"

class DeepLearning():
    model = tf.keras.models.Sequential()
    
    EPSILON = 0.1
    ACTIONS = [0, 1, 2, 3]
    y_min, y_max, x_min, x_max  = 25, 195, 20, 140
    input_shape = (y_max-y_min, x_max-x_min)
    # y = 170, x = 120 | x*y = 20400

    # Quantidade de Neuronios em cada camada
    n_input_layer = (y_max-y_min)*(x_max-x_min)
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
    
    def load_data(self):
        load_observation_list = np.load(PATH + '/npz/observation_list.npz')
        load_action_list = np.load(PATH + '/npz/action_list.npz')

        # print(type(load_observation_list))
        # print(type(load_action_list))
        # print(type(load_observation_list.f.arr_0))
        # print(load_observation_list.f.arr_0.size)
        # print(load_observation_list.f.arr_0)

        self.observation_list = load_observation_list.f.arr_0
        self.action_list = load_action_list.f.arr_0


    def train(self):
        print(self.model.summary())

        self.load_data()
        history = self.model.fit(self.observation_list, self.action_list, epochs=1, batch_size=10)
        self.plot_history(history)

        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        results = self.model.evaluate(self.observation_list, self.action_list, batch_size=32)
        print('test loss, test acc:', results)


    def predict(self, observation, reward):
        processed_observation = self.gray_crop(observation)
        tridimensional_data = np.expand_dims(processed_observation, axis=0)

        actions = self.model.predict(x=tridimensional_data)
        # print(actions)

        selected_action = np.argmax(actions)
        selected_action = self.optional_policy(selected_action)

        # print(selected_action)
        # print(actions[0, selected_action])

        return selected_action

    def optional_policy(self, randomic_action):
        rand_val = np.random.random()
        if rand_val < self.EPSILON:
            randomic_action = np.random.randint(0, len(self.ACTIONS))
        
        return randomic_action
        # return opt_policy, actions[0, opt_policy]

    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2)

    def plot_history(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(PATH + '/plt/accuracy.png')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(PATH + '/plt/loss.png')
        plt.show()
    
    # def store_model_graph(self):
    #     from keras.utils import plot_model
    #     plot_model(self.model, to_file='model.png')

a = DeepLearning()

