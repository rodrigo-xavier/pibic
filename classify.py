import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import math


WIDTH, HEIGHT = 50, 50
IMG_PATH = "../.database/pibic/pygame/img/"
NPZ_PATH = "../.database/pibic/pygame/npz/"
MODEL_PATH = "../.database/pibic/pygame/model/"


class BubblesClassifier():
    model = tf.keras.models.Sequential()

    ACTIONS = [0, 1]
    MAP_ACTIONS = {0: "quadrado", 1: "circulo"}

    input_shape = (WIDTH, HEIGHT)
    dim_input = (WIDTH)*(HEIGHT)
    dim_output = len(ACTIONS)
    nb_units = 5
    n_hidden_layer = round(math.sqrt((dim_input*dim_output)))

    def __init__(self):
        self.model.add(
            layers.SimpleRNN(
                self.n_hidden_layer,
                input_shape=(self.input_shape), 
                return_sequences=True,
                #units=self.nb_units
            )
        )
        self.model.add(
            layers.Dense(
                self.dim_output,
                activation='sigmoid'
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
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    
    def prepare_data(self):
        bubbles = np.load(NPZ_PATH+"bubbles.npz")
        _bubbles = bubbles.f.arr_0
        
        squares = np.zeros((int(_bubbles.shape[0]/2),), dtype=int)
        circles = np.ones((int(_bubbles.shape[0]/2),), dtype=int)

        self.input = np.reshape(_bubbles, _bubbles.shape)
        self.output = np.concatenate((squares, circles), axis=None)


    def train(self, epochs, batch_size):
        history = self.model.fit(self.input, self.output, epochs=epochs, batch_size=batch_size)
        self.save_model()
        self.plot_history(history)


    def predict(self, observation, reward):
        actions = self.model.predict(x=self.another_bubble)
        print(actions)

        selected_action = np.argmax(actions)

        # print(selected_action)
        # print(actions[0, selected_action])

        return selected_action
    
    def plot(self):
        pass


    def save_model(self):
        model_json = self.model.to_json()

        with open(MODEL_PATH+"model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(MODEL_PATH+"model.h5")
        print("Saved model to disk")


    def load_model(self):
        self.model = tf.keras.models.load_model(MODEL_PATH+"model.h5")


    def show_network(self):
        from ann_visualizer.visualize import ann_viz;

        # fix random seed for reproducibility
        np.random.seed(7)

        # load json and create model
        json_file = open(MODEL_PATH+'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        _model = tf.keras.models.model_from_json(loaded_model_json)
        
        # load weights into new model
        _model.load_weights(MODEL_PATH+"model.h5")
        ann_viz(_model, title="Artificial Neural network - Model Visualization")


a = BubblesClassifier()
a.prepare_data()
a.train(epochs=2, batch_size=32)
a.show_network()