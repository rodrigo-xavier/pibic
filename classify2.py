import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import SimpleRNN
import math
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
import graphviz
from interface import implements, Interface
import matplotlib.pyplot as plt



WIDTH, HEIGHT = 50, 50
IMG_PATH = "../.database/pibic/pygame/img/"
NPZ_PATH = "../.database/pibic/pygame/npz/"


pygame = np.load(NPZ_PATH+"bubbles.npz")
pygame = pygame.f.arr_0



target = []
for i in range(50):
    zero_um = (0,1)
    target.append(zero_um)

for i in range(50):
    um_zero = (1,0)
    target.append(um_zero)


x = np.reshape(pygame, (100, 50, 50))
y = np.array(target)





n_input_layer = 1000
n_output_layer = 1
n_hidden_layer = round(math.sqrt((n_input_layer*n_output_layer)))
print("nro de neurônios na hidden layer:", n_hidden_layer)






model=Sequential()
model.add(SimpleRNN(n_hidden_layer, 
                    input_shape=(None, 1),
                    kernel_initializer='random_normal'))
model.add(Dense(2, activation='sigmoid'))
# model.compile(loss = 'mse', optimizer = 'rmsprop')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()




plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


history = model.fit(x, y, epochs = 20, batch_size = 32, callbacks=[es])





# summarize history for accuracy
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='test')






# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")







test = np.reshape(test, (40, 1000, 1))





predictions = model.predict(test)
for i in range(len(test)):
    print(predictions[i], 'expected', test_target[i])
