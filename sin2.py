import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def sina(t=100, cycle=2):
    x = np.arange(0, cycle*t)
    return np.sin(2.0 * np.pi * x / t)

def sinb(t=100, cycle=2):
    x = np.arange(0, cycle*t)
    return np.sin(4.0 * np.pi * x / t)
 
def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def gen_wave(rawdata, inputlen=50):    
    input=[]
    target=[]
    for i in range(0, len(rawdata) - inputlen):  
        input.append( rawdata[i:i+inputlen] )  
        target.append( rawdata[i+inputlen] )  
    
    X = np.array(input).reshape(len(input), inputlen, 1) 
    Y = np.array(target).reshape(len(input), 1) 
    
    x, val_x, y, val_y = train_test_split(X, Y, test_size=int(len(X) * 0.2), shuffle=False)

    return x, y, val_x, val_y, rawdata

def model(x, y, val_x, val_y, n_in=1, inputlen=50):
    n_hidden = 20
    n_out = 1
    epochs = 50
    batch_size = 10
    
    model=Sequential()
    model.add(SimpleRNN(n_hidden, input_shape=(inputlen, n_in), kernel_initializer='random_normal'))
    model.add(Dense(n_out, kernel_initializer='random_normal'))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
    model.fit(x, y, batch_size=batch_size, epochs=epochs)
    # model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))

    return model

def predict(model, x, n_in=1, inputlen=50):
    in_ = x[:1]
    
    predicted = [None for _ in range(inputlen)] 
    
    for _ in range(len(rawdata) - inputlen):
        out_ = model.predict(in_) 
        in_ = np.concatenate( (in_.reshape(inputlen, n_in)[1:], out_), axis=0 ).reshape(1, inputlen, n_in)
        predicted.append(out_.reshape(-1))
    
    return predicted

def plot_predicted_wave(rawdata, predicted, x):
    plt.title('Predict sin wave')   
    plt.plot(rawdata, label="original")
    plt.plot(predicted, label="predicted")
    plt.plot(x[0], label="input")
    plt.legend()
    plt.show()


x, y, val_x, val_y, rawdata = gen_wave(noisy(sina()))
# x, y, val_x, val_y, rawdata = gen_wave(sina())

# plt.plot(sina())
# plt.show()
# plt.plot(noisy(sina()))
# plt.show()

plt.plot(y, label="training")
# plt.plot(val_y, label="validate")
plt.title('Target Values')
plt.legend()
plt.show()

m = model(x, y, val_x, val_y)
p = predict(m, x)
plot_predicted_wave(rawdata, p, x)





x, y, val_x, val_y, rawdata = gen_wave(noisy(sinb()))
# x, y, val_x, val_y, rawdata = gen_wave(sinb())

# plt.plot(sinb())
# plt.show()
# plt.plot(noisy(sinb()))
# plt.show()

plt.plot(y, label="training")
# plt.plot(val_y, label="validate")
plt.title('Target Values')
plt.legend()
plt.show()

p = predict(m, x)
plot_predicted_wave(rawdata, p, x)