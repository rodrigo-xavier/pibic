import numpy as np
import cv2
import os
import shutil
from game import BubblesGame

class AIData:
    tensor = []

    def __init__(self, IMG_PATH, NPZ_PATH, SURFACE_COLOR, FPS, CIRCLE_BUBBLES, SQUARE_BUBBLES, BUBBLES_COLOR, BUBBLES_RADIUS, WIDTH, HEIGHT, TRAJECTORY, TRAJECTORY_RADIUS, INCREASE_RADIUS):
        self.bubbles = BubblesGame(
            SURFACE_COLOR=SURFACE_COLOR,
            FPS=FPS,
            CIRCLE_BUBBLES=CIRCLE_BUBBLES,
            SQUARE_BUBBLES=SQUARE_BUBBLES,
            BUBBLES_COLOR=BUBBLES_COLOR,
            BUBBLES_RADIUS=BUBBLES_RADIUS,
            WIDTH=WIDTH,
            HEIGHT=HEIGHT,
            TRAJECTORY=TRAJECTORY,
            TRAJECTORY_RADIUS=TRAJECTORY_RADIUS,
        )
        
        self.img_path = IMG_PATH
        self.npz_path = NPZ_PATH
        self.trajectory = TRAJECTORY
        self.width = WIDTH
        self.height = HEIGHT

        self.circle_bubbles = CIRCLE_BUBBLES
        self.square_bubbles = SQUARE_BUBBLES
        self.increase_radius = INCREASE_RADIUS
        self.trajectory_radius = TRAJECTORY_RADIUS

    
    def reset_folder(self):
        try:
            shutil.rmtree(self.img_path)
            shutil.rmtree(self.npz_path)
        except OSError as e:
            print("Error: %s" % (e.strerror))
        try:
            os.mkdir(self.img_path)
            os.mkdir(self.npz_path)
        except OSError:
            print ("Creation of the directory failed")
    

    def play_game(self, frames, save):
        if self.increase_radius and self.trajectory != 'random':
            for i in range(1, self.trajectory_radius):
                self.bubbles.bubbles[0].trajectory_radius = i
                self.bubbles.run(i, self.img_path, frames, save)
        else:
            self.bubbles.run(0, self.img_path, frames, save)


    # 120 == Frames per lap
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
                        np.savez_compressed(self.npz_path + self.trajectory + "_" + "bola" + "_" + str(int(counter/120)) + ".npz", self.tensor)
                    if self.square_bubbles != 0:
                        np.savez_compressed(self.npz_path + self.trajectory + "_" + "quadrado" + "_" + str(int(counter/120)) + ".npz", self.tensor)
                    self.tensor = []

    def img_processing(self, img_path):
        img = cv2.imread(img_path, 0) # Convert to grayscale
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype('float32')
        img /= 255
        return img