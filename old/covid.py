import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import show_array_as_img, store_csv, store_png, store_npz, arguments
import matplotlib.pyplot as plt

import math


# Input

DATE = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110)
NEW_CASES = (1,0,0,1,0,0,0,1,4,6,6,6,0,9,18,25,21,23,79,34,57,137,193,283,224,418,345,310,232,482,502,486,353,323,1138,1117,1076,1146,1222,852,926,1661,2210,1930,1781,1089,1442,1261,1832,3058,2105,3257,2917,2055,1927,2498,2678,3735,3503,5514,3379,4613,5385,6276,7218,5919,5097,4751,6633,6935,10503,9888,10222,10611,6760,5632,9258,11385,13944,15305,14919,7938,13140,17408,19951,18508,20803,16508,15813,11687,16324,20599,26417,26928,33274,15760,12247,28936,28633,30925,30830,27075,18912,15654,32091,32913,30412,25982,21704,17110)
TOTAL_CASES = (1,1,1,2,2,2,2,3,7,13,19,25,25,34,52,77,98,121,200,234,291,428,621,904,1128,1546,1891,2201,2433,2915,3417,3903,4256,4579,5717,6834,7910,9056,10278,11130,12056,13717,15927,17857,19638,20727,22169,23430,25262,28320,30425,33682,36599,38654,40581,43079,45757,49492,52995,58509,61888,66501,71886,78162,85380,91299,96396,101147,107780,114715,125218,135106,145328,155939,162699,168331,177589,188974,202918,218223,233142,241080,254220,271628,291579,310087,330890,347398,363211,374898,391222,411821,438238,465166,498440,514200,526447,555383,584016,614941,645771,672846,691758,707412,739503,772416,802828,828810,850514,867624)
NEW_DEATHS = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,2,5,7,7,9,12,11,20,15,22,22,23,42,40,58,60,73,54,67,114,133,141,115,68,99,105,204,204,188,217,206,115,113,166,165,407,357,346,189,338,474,449,435,428,395,301,296,600,614,611,749,732,496,396,881,749,844,824,816,485,674,1179,888,1188,999,967,653,807,1039,1086,1156,1124,956,480,623,1262,1349,1473,1005,904,525,679,1272,1274,1239,909,892,612)
TOTAL_DEATHS = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,4,6,11,18,25,34,46,57,77,92,114,136,159,201,241,299,359,432,486,553,667,800,941,1056,1124,1223,1328,1532,1736,1924,2141,2347,2462,2575,2741,2906,3313,3670,4016,4205,4543,5017,5466,5901,6329,6724,7025,7321,7921,8535,9146,9895,10627,11123,11519,12400,13149,13993,14817,15633,16118,16792,17971,18859,20047,21046,22013,22666,23473,24512,25598,26754,27878,28834,29314,29937,31199,32548,34021,35026,35930,36455,37134,38406,39680,40919,41828,42720,43332)

date = np.asarray(DATE)
new_cases = np.asarray(NEW_CASES)
total_cases = np.asarray(TOTAL_CASES)
new_deaths = np.asarray(NEW_DEATHS)
total_deaths = np.asarray(TOTAL_DEATHS)

df = pd.DataFrame(total_cases)
data = df.values

# Define train and test

N = int(len(date))
T = int(len(date) * 0.8)

train, test = data[0:T,:], data[T:N,:]

# Convert to Matrix

def convertToMatrix(data):
    X, Y = [], []
    
    for i in range(len(data)-1):
        d=i+1
        X.append(data[i:d,])
        Y.append(data[d,])
    
    return np.array(X), np.array(Y)

trainX,trainY = convertToMatrix(train)
testX,testY = convertToMatrix(test)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Define Model

model = tf.keras.models.Sequential()
    
model.add(layers.SimpleRNN(units=32, input_shape=(1,1), activation="relu"))
model.add(layers.Dense(8, activation="relu")) 
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()

model.fit(trainX, trainY, epochs=200, batch_size=16, verbose=2)
trainPredict = model.predict(trainX)
testPredict= model.predict(testX)
predicted=np.concatenate((trainPredict,testPredict),axis=0)

trainScore = model.evaluate(trainX, trainY, verbose=0)
print(trainScore)

index = df.index.values
plt.plot(index,df)
index = index[0:len(index)-2] # Gambiarra
plt.plot(index,predicted)
plt.axvline(df.index[T], c="r")
plt.show()