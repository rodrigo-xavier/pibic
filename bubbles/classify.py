import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import math
import os



WIDTH, HEIGHT = 50, 50
IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"
TEST_NPZ_PATH = "../../.database/pibic/pygame/npz_test/"
MODEL_PATH = "../../.database/pibic/pygame/model/"


class BubblesClassifier():
    model = tf.keras.models.Sequential()

    ACTIONS = [0, 1]
    MAP_ACTIONS = {0: "quadrado", 1: "circulo"}

    input_shape = (WIDTH, HEIGHT)
    dim_input = (WIDTH)*(HEIGHT)
    dim_output = len(ACTIONS)
    n_hidden_layer = round(math.sqrt((dim_input*dim_output)))

    def __init__(self):
        self.model.add(
            layers.SimpleRNN(
                units=self.n_hidden_layer,
                input_shape=self.input_shape,
                activation='tanh',
                kernel_initializer='random_uniform',
            )
        )
        self.model.add(
            layers.Dense(
                self.dim_output,
                activation='sigmoid'
            )
        )
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    
    def prepare_data(self):
        n = len(os.listdir(NPZ_PATH))
        
        for f in os.listdir(NPZ_PATH):

            file_name = f.split('_')
            if 'circle' in file_name:
                np.load(f)

            bubbles = np.load(NPZ_PATH+"bubbles.npz")

            _bubbles = bubbles.f.arr_0

            self.input = _bubbles
            
            squares = np.zeros((int(_bubbles.shape[0]/2),2), dtype=int)
            circles = np.ones((int(_bubbles.shape[0]/2),2), dtype=int)

            squares[:, [-1]] = 1
            circles[:, [-1]] = 0

            self.output = np.append(squares, circles, axis=0)


    def train(self, epochs, batch_size):
        self.history = self.model.fit(self.input, self.output, epochs=epochs, batch_size=batch_size)

        scores = self.model.evaluate(self.input_test, self.output_test)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

        self.save_model()


    def predict(self, observation, reward):
        actions = self.model.predict(x=self.input)
        print(actions)

        selected_action = np.argmax(actions)

        # print(selected_action)
        # print(actions[0, selected_action])

        return selected_action
    

    def plot_graph(self):
        plt.plot(self.history.history['accuracy'])
        # plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        # plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


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
        ann_viz(self.model, view=True, title="Artificial Neural network - Model Visualization")
    
    def plot_network(self):
        from keras.utils.vis_utils import plot_model
        import graphviz
        from interface import implements, Interface

        path = MODEL_PATH + "model_plot.png"
        plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True)
        print("Saved network graph to disk")


a = BubblesClassifier()
a.prepare_data()
a.train(epochs=300, batch_size=32)
a.plot_network()
a.plot_graph()
# a.show_network()