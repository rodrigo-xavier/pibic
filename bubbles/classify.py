import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import math
import os


WIDTH, HEIGHT = 50, 50
PATH = "../../.database/pibic/pygame/"


class BubblesClassifier():
    model = tf.keras.models.Sequential()

    dict_of_fit = {}
    dict_of_predict = {}

    input_fee = 0.8

    ACTIONS = [0, 1]
    MAP_ACTIONS = {"quadrado": [0, 1], "circulo": [1, 0]}

    # input_shape = (120, WIDTH, HEIGHT)
    input_shape = (120, 2500)
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

    
    def load_and_prepare_data(self, *args):
        arrays = []
        self.dict_of_fit.update({"circle": [], "square": []})
        self.dict_of_predict.update({"circle": [], "square": []})

        for subpath in args:
            path = PATH + subpath
            n = len(os.listdir(path))

            trajectory_type = subpath.split("/")[1]

            for i, file in enumerate(os.listdir(path)):
                if file.find(".npz") != -1:
                    lap = np.load(path + file)
                    redimensioned = np.reshape(lap.f.arr_0, (1, 120, 2500))
                    arrays.append(redimensioned)
                    # arrays.append(lap.f.arr_0)
                    
                    if i == int(n * self.input_fee):
                        self.dict_of_fit[trajectory_type].append(np.concatenate(arrays))
                        arrays = []
                    elif i == n-1:
                        self.dict_of_predict[trajectory_type].append(np.concatenate(arrays))
                        arrays = []
        
        self.concat_data()
        self.build_expected_output()
    
    def concat_data(self):
        circle = np.concatenate((self.dict_of_fit["circle"]))
        square = np.concatenate((self.dict_of_fit["square"]))
        self.input_fit = np.concatenate((circle, square))

        print(self.input_fit.shape)

        circle = np.concatenate((self.dict_of_predict["circle"]))
        square = np.concatenate((self.dict_of_predict["square"]))
        self.input_predict = np.concatenate((circle, square))
    
    def build_expected_output(self):
        circles = np.ones((int(self.input_fit.shape[0]/(2)),2), dtype=int)
        squares = np.zeros((int(self.input_fit.shape[0]/(2)),2), dtype=int)

        circles[:, [-1]] = 0
        squares[:, [-1]] = 1

        self.output_fit = np.append(circles, squares, axis=0)

        circles = np.ones((int(self.input_predict.shape[0]/(2)),2), dtype=int)
        squares = np.zeros((int(self.input_predict.shape[0]/(2)),2), dtype=int)

        circles[:, [-1]] = 0
        squares[:, [-1]] = 1

        self.output_predict = np.append(circles, squares, axis=0)

    def fit(self, epochs, batch_size):
        self.history = self.model.fit(self.input_fit, self.output_fit, epochs=epochs, batch_size=batch_size)

        # scores = self.model.evaluate(self.dict_of_predict, self.output_predict)
        # print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

        self.save_model()


    def predict(self, observation, reward):
        actions = self.model.predict(x=self.input_predict)
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
        plt.legend(['fit', 'predict'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        # plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['fit', 'predict'], loc='upper left')
        plt.show()


    def save_model(self):
        model_json = self.model.to_json()

        with open(PATH + "model/model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(PATH + "model/model.h5")
        print("Saved model to disk")


    def load_model(self):
        self.model = tf.keras.models.load_model(PATH + "model/model.h5")


    def show_network(self):
        from ann_visualizer.visualize import ann_viz;

        # fix random seed for reproducibility
        np.random.seed(7)

        # load json and create model
        json_file = open(PATH + 'model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        _model = tf.keras.models.model_from_json(loaded_model_json)
        
        # load weights into new model
        _model.load_weights(PATH + "model/model.h5")
        ann_viz(self.model, view=True, title="Artificial Neural network - Model Visualization")
    
    def plot_network(self):
        from keras.utils.vis_utils import plot_model
        import graphviz
        from interface import implements, Interface

        path = PATH + "model/model_plot.png"
        plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True)
        print("Saved network graph to disk")


a = BubblesClassifier()
# a.load_and_prepare_data("pack/circle/1/", "pack/square/1/", "pack/circle/2/", "pack/square/2/")
a.load_and_prepare_data("pack/circle/1/", "pack/square/1/")
a.fit(epochs=30, batch_size=32)
# a.predict()
a.plot_network()
a.plot_graph()
a.show_network()