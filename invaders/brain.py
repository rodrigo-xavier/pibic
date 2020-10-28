from data import DataPreProcessing
import math, copy
import numpy as np

class Neural():
    """
    docstring
    """

    error_rate = 0
    layer_1_values = list()
    layer_2_deltas = list()

    def __init__(self, **kwargs):
        self.input_neurons = (kwargs['y_max']-kwargs['y_min'])*(kwargs['x_max']-kwargs['x_min'])
        self.output_neurons = len(kwargs['ACTIONS'])
        self.hidden_neurons = round(math.sqrt((self.input_neurons*self.output_neurons)))
        self.alpha = kwargs['alpha']

        self.synapse_i_h = 2 * np.random.random_sample((self.input_neurons, self.hidden_neurons))  - 1
        self.synapse_h_h = 2 * np.random.random_sample((self.hidden_neurons, self.hidden_neurons)) - 1
        self.synapse_h_o = 2 * np.random.random_sample((self.hidden_neurons, self.output_neurons)) - 1

        self.synapse_i_h_update = np.zeros_like(self.synapse_i_h)
        self.synapse_h_h_update = np.zeros_like(self.synapse_h_h)
        self.synapse_h_o_update = np.zeros_like(self.synapse_h_o)

        self.layer_1_values.append(np.zeros(self.hidden_neurons))
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)
    
    def funcname(self):
        pass
    
    def show(self, predicted):
        out = 0

        print ("Error: " + str(self.error_rate))
        print ("Pred: " + str(d))
        print ("True: " + str(c))
        print "------------"
    
    def plot(self):
        pass
    
    def load(self):
        pass

    def save(self):
        pass


class LSTM(Neural, DataPreProcessing):
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    alpha = 0.1
    
    ACTIONS = [0, 1, 2, 3]

    def __init__(self, **kwargs):
        kwargs.update({'y_min':self.y_min, 'y_max':self.y_max, 'x_min':self.x_min, 'x_max':self.x_max})
        kwargs.update({'alpha':self.alpha})
        super().__init__(**kwargs)
    
    def train(self, observation, reward):
        observation = self.gray_crop(observation)

        # input and output
        layer_0 = observation
        output = self.ACTIONS

        # hidden layer (input ~+ prev_hidden)
        layer_1 = self.sigmoid(np.dot(layer_0, self.synapse_i_h) + np.dot(self.layer_1_values[-1], self.synapse_h_h))

        # output layer (new binary representation)
        layer_2 = self.sigmoid(np.dot(layer_1, self.synapse_h_o))

        # did we miss?... if so, by how much?
        layer_2_error = output - layer_2
        self.layer_2_deltas.append((layer_2_error)*self.sigmoid_output_to_derivative(layer_2))
        self.error_rate += np.abs(layer_2_error[0])

        # decode estimate so we can print it out
        out = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        self.layer_1_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(self.hidden_neurons)

        # backpropagating

        layer_0 = observation
        layer_1 = self.layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        self.synapse_i_h_update += layer_0.T.dot(layer_1_delta)
        self.synapse_h_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        self.synapse_h_o_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)

        future_layer_1_delta = layer_1_delta

        self.synapse_i_h += self.synapse_i_h_update * self.alpha
        self.synapse_h_h += self.synapse_h_h_update * self.alpha   
        self.synapse_h_o += self.synapse_h_o_update * self.alpha

        self.synapse_i_h_update *= 0
        self.synapse_h_h_update *= 0
        self.synapse_h_o_update *= 0

