import sys
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import zipfile

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop

base_path = 'database/flowers/'
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
img_width, img_height = 256, 256

def load_images():
    fnames = []
    images = []

    for category in categories:
        flower_folder = os.path.join(base_path, category)
        file_names = os.listdir(flower_folder)
        full_path = [os.path.join(flower_folder, file_name) for file_name in file_names]
        fnames.append(full_path)

    for names in fnames:
        one_category_images = [cv2.imread(name) for name in names if (cv2.imread(name)) is not None]
        images.append(one_category_images)

    print('number of images for each category:', [len(f) for f in images])

    return images


def minimal_shape(images):
    for i,imgs in enumerate(images):
        shapes = [img.shape for img in imgs]
        widths = [shape[0] for shape in shapes]
        heights = [shape[1] for shape in shapes]
        print('%d,%d is the min shape for %s' % (np.min(widths), np.min(heights), categories[i]))

def cvtRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

def show_flowers_by_type(images):
    plt.figure(figsize=(15,10))
    for i, imgs in enumerate(images):
        plt.subplot(2,3,i+1)
        idx = np.random.randint(len(imgs))
        plt.imshow(cvtRGB(imgs[idx]))
        plt.grid('off')
        plt.title(categories[i]+' '+str(idx))
    plt.show()


def resize_images(images):
    resized_images = []
    
    for i,imgs in enumerate(images):
        resized_images.append([
            cv2.resize(
                img, 
                (img_width, img_height), 
                interpolation = cv2.INTER_CUBIC) for img in imgs
        ])
    
    return resized_images


# Split dataset to 80% of training and 20% of validation
def split_images(resized_images):
    train_images = []
    val_images = []

    for imgs in resized_images:
        train, test = train_test_split(imgs, train_size=0.8, test_size=0.2)
        train_images.append(train)
        val_images.append(test)
    
    return train_images, val_images


def create_label(train_images, val_images):
    len_train_images = [len(imgs) for imgs in train_images]
    train_categories = np.zeros((np.sum(len_train_images)), dtype='uint8')

    len_val_images = [len(imgs) for imgs in val_images]
    val_categories = np.zeros((np.sum(len_val_images)), dtype='uint8')

    for i in range(5):
        if i is 0:
            train_categories[:len_train_images[i]] = i
        else:
            train_categories[np.sum(len_train_images[:i]):np.sum(len_train_images[:i+1])] = i
            
    for i in range(5):
        if i is 0:
            val_categories[:len_val_images[i]] = i
        else:
            val_categories[np.sum(len_val_images[:i]):np.sum(len_val_images[:i+1])] = i
        
    return train_categories, val_categories


def img_to_np(train_images, val_images):
    tmp_train_imgs = []
    tmp_val_imgs = []

    for imgs in train_images:
        tmp_train_imgs += imgs
    for imgs in val_images:
        tmp_val_imgs += imgs

    train_images = np.array(tmp_train_imgs)
    val_images = np.array(tmp_val_imgs)

    return train_images, val_images


def shuffle_dataset(train_images, val_images, train_categories, val_categories):
    train_data = train_images.astype('float32')
    val_data = val_images.astype('float32')
    train_labels = np_utils.to_categorical(train_categories, len(categories))
    val_labels = np_utils.to_categorical(val_categories, len(categories))

    seed = 100
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)
    np.random.seed(seed)
    np.random.shuffle(val_data)
    np.random.seed(seed)
    np.random.shuffle(val_labels)

    train_data = train_data[:3400]
    train_labels = train_labels[:3400]
    val_data = val_data[:860]
    val_labels = val_labels[:860]
    print('shape of train data:', train_data.shape)
    print('shape of train labels:', train_labels.shape)
    print('shape of val data:', val_data.shape)
    print('shape of val labels:', val_labels.shape)

    return train_data, train_labels, val_data, val_labels


def create_model_from_scratch(train_data):

    """
     train from scratch
    """

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=train_data.shape[1:], activation='relu', name='conv_1'))
    model.add(Conv2D(32, (3,3), activation='relu', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_1'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu', name='conv_3'))
    model.add(Conv2D(64, (3,3), activation='relu', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_2'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', name='conv_5'))
    model.add(Conv2D(128, (3,3), activation='relu', name='conv_6'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_3'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(len(categories), name='output'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # optimizer=RMSprop(lr=0.001)
    
    return model


def predict_one_image(img, model):
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    img = np.reshape(img, (1, img_width, img_height, 3))
    img = img/255.
    pred = model.predict(img)
    class_num = np.argmax(pred)
    return class_num, np.max(pred)


def predict_val(val_data, model):
    val_input = np.reshape(val_data, (1, img_width, img_height, 3))
    val_input = val_input/255.
    pred = model.predict(val_input)
    class_num = np.argmax(pred)
    return class_num, np.max(pred)


def plot_model_history(model_name, history, epochs):
    print(model_name)
    plt.figure(figsize=(15, 5))

    # summarize history for accuracy
    plt.subplot(1, 2 ,1)
    plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
    plt.plot(np.arange(1, len(history['val_acc'])+1), history['val_acc'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['loss'])+1), history['loss'], 'r')
    plt.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Loss vs. Validation Loss')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='best')


    plt.show()


def return_name(label_arr):
  idx = np.where(label_arr == 1)
  return idx[0][0]



img = load_images()
img = resize_images(img)
train_images, val_images = split_images(img)

train_categories, val_categories = create_label(train_images, val_images)
train_images, val_images = img_to_np(train_images, val_images)

train_data, train_labels, val_data, val_labels = shuffle_dataset(train_images, val_images, train_categories, val_categories)

model_scratch = create_model_from_scratch(train_data)
model_scratch.summary()



# Parameters
batch_size = 32
epochs1 = 50
epochs2 = 10
epochs3 = 30


# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(
    rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow(
    train_data,
    train_labels,
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    val_data,
    val_labels,
    batch_size=batch_size
)


start = time.time()

model_scratch_info = model_scratch.fit_generator(
    generator=train_generator, 
    steps_per_epoch=len(train_data)/batch_size,   # -> 106 # images 3392 = steps * batch_size = 106 * 32 
    epochs=epochs1, 
    validation_steps=len(val_data)/batch_size, # -> 26 # images 832 = steps * batch_size = 26 * 32
    validation_data=val_generator, 
    verbose=2
)

end = time.time()
duration = end - start
print ('\n model_scratch took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs1) )

plot_model_history('model_scratch', model_scratch_info.history, epochs1)


test_img = cv2.imread('database/flowers/which_type/1a179.jpg')
pred, probability = predict_one_image(test_img, model_scratch)
print('%s %d%%' % (categories[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(cvtRGB(test_img))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()



print("Model trained from Scrach")
plt.figure(figsize=(15,15))
for i in range(9):
  
  idx = np.random.randint(860)
  
  ax = plt.subplot(3,3,i+1)
  plt.imshow(cvtRGB(val_data.astype('uint8')[idx]))
  category_idx = return_name(val_labels[idx])
  
  pred, prob = predict_val(val_data[idx], model_scratch)
  plt.title('True: %s || Predict: %s %d%%' % (categories[category_idx], categories[pred], round(prob, 2)*100))
  plt.grid(False)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  
plt.show()


my_model.save('database/model/')