import numpy as np
import cv2
import os

class AIData:
    tensor = []

    def __init__(self, IMG_PATH, NPZ_PATH):
        self.img_path = IMG_PATH
        self.npz_path = NPZ_PATH

    def img2npz(self):
        # from utils import show_array_as_img
        for f in os.listdir(self.img_path):
            if f.find(".png") != -1:
                img = self.img_processing("{}/{}".format(self.img_path, f))
                # show_array_as_img(img, 'gray')
                self.tensor.append(img)

        apart = int(len(self.tensor)*0.8)

        np.savez_compressed(self.npz_path + "bubbles_train.npz", self.tensor[:apart])
        np.savez_compressed(self.npz_path + "bubbles_test.npz", self.tensor[apart:])

    def img_processing(self, img_path):
        img = cv2.imread(img_path, 0) # Convert to grayscale
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype('float32')
        img /= 255
        return img
