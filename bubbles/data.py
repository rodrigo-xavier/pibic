import numpy as np
import cv2
import os
import shutil

class AIData:
    tensor = []

    def __init__(self, IMG_PATH, NPZ_PATH, TRAJECTORY, WIDTH, HEIGHT, CIRCLE_BUBBLES, SQUARE_BUBBLES):
        self.img_path = IMG_PATH
        self.npz_path = NPZ_PATH
        self.trajectory = TRAJECTORY
        self.width = WIDTH
        self.height = HEIGHT

        self.circle_bubbles = CIRCLE_BUBBLES
        self.square_bubbles = SQUARE_BUBBLES
    
    def reset_folder(self):
        try:
            shutil.rmtree(self.img_path)
        except OSError as e:
            print("Error: %s : %s" % (self.img_path, e.strerror))
        try:
            os.mkdir(self.img_path)
        except OSError:
            print ("Creation of the directory %s failed" % self.img_path)

    def img2npz(self):
        counter = 0

        if self.trajectory == 'random':
            for f in os.listdir(self.img_path):
                if f.find(".png") != -1:
                    img = self.img_processing("{}/{}".format(self.img_path, f))
                    # from utils import show_array_as_img
                    # show_array_as_img(img, 'gray')
                    self.tensor.append(img)

            apart = int(len(self.tensor)*0.8)

            np.savez_compressed(self.npz_path + self.trajectory +"_train.npz", self.tensor[:apart])
            np.savez_compressed(self.npz_path + self.trajectory +"_test.npz", self.tensor[apart:])

        else:
            for f in os.listdir(self.img_path):
                counter += 1

                if f.find(".png") != -1:
                    img = self.img_processing("{}/{}".format(self.img_path, f))
                    self.tensor.append(img)

                if counter % 120 == 0:
                    if self.circle_bubbles != 0:
                        np.savez_compressed(self.npz_path + self.trajectory + "_" + "bola" + str(int(counter/120)) + ".npz", self.tensor)
                    if self.square_bubbles != 0:
                        np.savez_compressed(self.npz_path + self.trajectory + "_" + "quadrado" + str(int(counter/120)) + ".npz", self.tensor)

    def img_processing(self, img_path):
        img = cv2.imread(img_path, 0) # Convert to grayscale
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype('float32')
        img /= 255
        return img
