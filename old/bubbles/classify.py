import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import math


WIDTH, HEIGHT = 50, 50
IMG_PATH = "../.database/pibic/pygame/img/"
NPZ_PATH = "../.database/pibic/pygame/npz/"
TEST_NPZ_PATH = "../.database/pibic/pygame/npz_test/"
MODEL_PATH = "../.database/pibic/pygame/model/"


class BubblesClassifier():
    model = tf.keras.models.Sequential()

    ACTIONS = [0, 1]
    MAP_ACTIONS = {0: "quadrado", 1: "circulo"}

    input_shape = (WIDTH, HEIGHT)
    dim_input = (WIDTH)*(HEIGHT)
    dim_output = len(ACTIONS)
    # dim_output = 1
    n_hidden_layer = round(math.sqrt((dim_input*dim_output)))

    def __init__(self):
        self.model.add(
            layers.SimpleRNN(
                units=self.n_hidden_layer,
                input_shape=self.input_shape,
                activation='tanh',
                kernel_initializer='random_uniform',
                # input_shape=self.input_shape,
                # bias_initializer='zeros'
                # return_sequences=True,
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
        self.model.add(
            layers.Dense(
                self.dim_output,
                activation='sigmoid'
            )
        )
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    
    def prepare_data(self):
        bubbles = np.load(NPZ_PATH+"bubbles.npz")
        _bubbles = bubbles.f.arr_0
        bubble_test = np.load(TEST_NPZ_PATH+"bubbles.npz")
        self.input_test = bubble_test.f.arr_0
        
        # zeros = np.zeros((int(_bubbles.shape[0]/2),2), dtype=int)
        # ones = np.ones((int(_bubbles.shape[0]/2),), dtype=int)

        # squares = zeros
        # squares[:, [-1]]=1
        # circles = zeros
        # circles[:, [0]]=1

        # self.input = np.reshape(_bubbles, _bubbles.shape)
        # self.input = np.reshape(_bubbles, _bubbles.shape)
        self.input = _bubbles
        # self.input = _bubbles.shape()
        # self.output = np.concatenate((squares, circles), axis=None)


        target = []
        for i in range(100):
            zero_um = (0,1)
            target.append(zero_um)

        for i in range(100):
            um_zero = (1,0)
            target.append(um_zero)
        
        self.output = np.array(target)
        self.output_test = np.array(target)


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
a.train(epochs=3000, batch_size=32)
a.plot_network()
a.plot_graph()
# a.show_network()