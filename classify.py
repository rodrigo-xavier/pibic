import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import math


WIDTH, HEIGHT = 50, 50
IMG_PATH = "../.database/pibic/pygame/img/"
NPZ_PATH = "../.database/pibic/pygame/npz/"


class BubblesClassifier():
    model = tf.keras.models.Sequential()

    input_shape = (WIDTH, HEIGHT)
    dim_input = (WIDTH)*(HEIGHT)
    dim_output = len(ACTIONS)
    nb_units = 5
    n_hidden_layer = round(math.sqrt((dim_input*dim_output)))

    def __init__(self):
        # self.model.add(
        #     layers.Flatten(
        #         input_shape=self.input_shape
        #     )
        # )
        self.model.add(
            layers.SimpleRNN(
                self.n_hidden_layer,
                input_shape=(None, self.dim_input), 
                return_sequences=True, 
                units=self.nb_units
            )
        )
        self.model.add(
            layers.Dense(
                self.dim_output,
                activation='softmax'
            )
        )
        # self.model.add(
        #     layers.TimeDistributed(
        #         layers.Dense(
        #             activation='sigmoid',
        #             units=self.dim_output
        #         )
        #     )
        # )
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
        pygame = np.load(NPZ_PATH+"bubbles.npz")
        self.pygame = pygame.f.arr_0

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


a = BubblesClassifier()
a.load_data()
a.train(epochs=10, batch_size=32)
a.load_another_data()
a.predict()
