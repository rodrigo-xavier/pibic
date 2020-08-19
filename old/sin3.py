import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#Generating Randon Data
t=np.arange(0, 1000)
x=np.sin(0.02*t)+2*np.random.rand(1000)
df=pd.DataFrame(x)
df.head()

#Splitting into Train and Test set
values = df.values
train, test = values[0:800,:], values[800:1000,:]

def convertToMatrix(data, step=4):
    X, Y = [], []
    for i in range(len(data)-step):
        d=i+step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

trainX,trainY =convertToMatrix(train,6)
testX,testY =convertToMatrix(test,6)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(SimpleRNN (units=32, input_shape=(1,6), activation='relu'))


model.add(Dense(8, activation="relu"))
model.add(Dense(1))
#Compiling the Code
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()
#Training the Model
model.fit(trainX, trainY, epochs=1, batch_size=500, verbose=2)
#Predicting with the Model
trainPredict = model.predict(trainX)
testPredict= model.predict(testX)
predicted=np.concatenate((trainPredict, testPredict),axis=0)


import matplotlib.pyplot as plt
plt.title('Predict sin wave')   
plt.plot(test, label="original")
plt.plot(predicted, label="predicted")
# plt.plot(x, label="input")
plt.legend()
plt.show()