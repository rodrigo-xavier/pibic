import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import show_array_as_img, store_csv, store_png, store_npz, arguments
import matplotlib.pyplot as plt

import math


# Randonly generate N
# def random_data(N):
#     t = np.arange(0,N)
#     x = np.sin(0.02*t)+2*np.random.rand(N)
#     dataframe = pd.DataFrame(x)
#     dataframe.head()
#     values = dataframe.values

#     return values, dataframe


# # Preparing N to insert in our model
# def reshaping(N, train, step, values):
#     train,test = values[0:train,:], values[train:N,:]

#     # add step elements into train and test
#     test = np.append(test,np.repeat(test[-1,],step))
#     train = np.append(train,np.repeat(train[-1,],step))

#     trainX,trainY = convertToMatrix(train,step)
#     testX,testY = convertToMatrix(test,step)
#     trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#     testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#     return train, trainX, trainY, testX


def prepare_data():
    f50 = np.load("f50.npy")
    f100 = np.load("f100.npy")
    f150 = np.load("f150.npy")
    f200 = np.load("f200.npy")
    f50_100 = np.load("f50_100.npy")

    f50_target, f100_target = [],[]
    for i in range(100):
        f50_target.append(0)
        f100_target.append(1)

    f50_100_target = []
    for i in range(100):
        f50_100_target.append(0)

    for i in range(100):
        f50_100_target.append(1)

    x = np.reshape(f50_100, (200, 1000, 1))
    y = np.array(f50_100_target)

    return x, y


def convertToMatrix(data):
    X, Y = [], []
    
    for i in range(len(data)-1):
        d=i+1
        X.append(data[i:d,])
        Y.append(data[d,])
    
    return np.array(X), np.array(Y)


#  Build model, create layers, train, predict and evaluate N.
def sinusoidal(x, y):
    # Init
    model = tf.keras.models.Sequential()
        
    model.add(layers.SimpleRNN(units=32, input_shape=(1000,1), activation="relu"))
    model.add(layers.Dense(8, activation="relu")) 
    model.add(layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.summary()

    # Train
    model.fit(x, y, epochs = 100, batch_size = 32)

    return model

def predict_f(model, x):
    # Predict
    predicted = model.predict(x)
    print(model.predict_classes(x))

    # Evaluate
    # trainScore = model.evaluate(x, y, verbose=0)
    # print(trainScore)

    return predicted


# Show graph of predicted N
def showgraph(dataframe, predicted):
    index = dataframe.index.values
    # plt.plot(index,dataframe)
    # plt.plot(predicted)
    plt.show()


# def run(N, train, step, sin, dataframe):
#     train, trainX, trainY, testX = reshaping(N, train, step, sin)
#     predicted = sinusoidal(trainX, trainY, testX, step)
#     showgraph(dataframe, train, predicted)


# Input
N = 1000
train = 800
step = 4
pi = 3.1415
f = 100

x, y = prepare_data()
model = sinusoidal(x, y)

t = np.arange(0,N)
x = np.sin(2*pi*f*t)+2*np.random.rand(N)
dataframe = pd.DataFrame(x)
x = np.reshape(x, (1, 1000, 1))

predicted = predict_f(model, x)
# showgraph(dataframe, predicted)