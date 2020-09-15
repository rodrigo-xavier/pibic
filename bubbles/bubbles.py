import pygame
from pygame.locals import *

import sys
import cv2
import numpy as np
import os
from circle import Circle
from square import Square
import math


# IMG_PATH = "../../.database/pibic/pygame/img/"
# NPZ_PATH = "../../.database/pibic/pygame/npz/"

IMG_PATH = "../../database/img/"
NPZ_PATH = "../../database/npz/"

class Bubbles:
    geometric = []
    tensor = []

    def __init__(self, SURFACE_COLOR=(0,0,0), FPS=60, CIRCLE_BUBBLES=1, SQUARE_BUBBLES=1, BUBBLES_COLOR=(255,255,255), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50, MOVEMENT_SHAPE='circular', TRAGETORY_RADIUS=12):
        
        self.surface = pygame.display.set_mode((WIDTH,HEIGHT))
        self.surface_color = SURFACE_COLOR
        self.fps = FPS
        self.bubbles_color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS
        self.width = WIDTH
        self.height = HEIGHT
        self.movement_shape = MOVEMENT_SHAPE
        self.tragetory_radius = TRAGETORY_RADIUS

        for i in range(CIRCLE_BUBBLES):
            self.geometric.append(Circle(self.surface, self.bubbles_color, self.radius, self.width, self.height, self.movement_shape, self.tragetory_radius))
        for i in range(SQUARE_BUBBLES):
            self.geometric.append(Square(self.surface, self.bubbles_color, self.radius, self.width, self.height, self.movement_shape, self.tragetory_radius))

    def move(self):
        for geometric in self.geometric:
            geometric.board_collision()
            if geometric.check_collision():
                geometric.elastic_collision(self.take_the_nearest(geometric))
            
            geometric.show()
    
    def circular_trajectory(self):
        
        '''if MOVEMENT_SHAPE == 'circular':
            if self.ang_idx == n_angles:
                self.ang_idx = 0
                self.n_loops = self.n_loops + 1

            x_center = radius * math.cos(self.angles[self.ang_idx])
            y_center = radius * math.sin(self.angles[self.ang_idx])

            self.ang_idx = self.ang_idx + 1

            self.x = CIRCULAR_CENTER[0] + x_center
            self.y = CIRCULAR_CENTER[1] + y_center

            print(self.n_loops)'''

    def take_the_nearest(self, geometric):
        distance = []

        for nearest in self.geometric:
            if nearest != geometric:
                distance.append(math.sqrt(((geometric.x-nearest.x)**2)+((geometric.y-nearest.y)**2)))

        return self.geometric[distance.index(min(distance))]

    def show(self):
        self.surface.fill(self.surface_color)
        self.move()
        pygame.display.flip()
        pygame.time.Clock().tick(self.fps)
    
    def save(self, n):
        file = IMG_PATH + str(n) + '.png'
        pygame.image.save(self.surface, file)

    def img2npz(self):
        # from utils import show_array_as_img
        for f in os.listdir(IMG_PATH):
            if f.find(".png") != -1:
                img = self.img_processing("{}/{}".format(IMG_PATH, f))
                # show_array_as_img(img, 'gray')
                self.tensor.append(img)

        apart = int(len(self.tensor)*0.8)

        np.savez_compressed(NPZ_PATH + "bubbles_train.npz", self.tensor[:apart])
        np.savez_compressed(NPZ_PATH + "bubbles_test.npz", self.tensor[apart:])

    def img2npz_trajetory(self):
        for f in os.listdir(IMG_PATH):
            if f.find(".png") != -1:
                img = self.img_processing("{}/{}".format(IMG_PATH, f))
                self.tensor.append(img)
        
        volta = len(os.listdir())/1320
        apart = int(volta*0.8)

        np.savez_compressed(NPZ_PATH + "bubbles_train.npz", self.tensor[:apart])
        np.savez_compressed(NPZ_PATH + "bubbles_test.npz", self.tensor[apart:])
    
    def img_processing(self, img_path):
        img = cv2.imread(img_path, 0) # Convert to grayscale
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype('float32')
        img /= 255
        return img

    def close(self):
        keystate = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or keystate[K_ESCAPE]:
                pygame.quit(); sys.exit()