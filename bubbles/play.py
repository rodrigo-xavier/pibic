import pygame
from pygame.locals import *

import sys
import cv2
import numpy as np
import os
from circle import Circle
from square import Square
import math


IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"


class Play:
    geometric = []
    tensor = []

    def __init__(self, SURFACE_COLOR=(0,0,0), FPS=60, CIRCLE_BUBBLES=1, SQUARE_BUBBLES=1, BUBBLES_COLOR=(255,255,255), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50):
        
        self.surface = pygame.display.set_mode((WIDTH,HEIGHT))
        self.surface_color = SURFACE_COLOR
        self.fps = FPS

        for i in range(CIRCLE_BUBBLES):
            self.geometric.append(
                Circle(
                    surface=self.surface,
                    surface_color=self.surface_color,
                    bubbles_color=BUBBLES_COLOR,
                    radius=BUBBLES_RADIUS,
                    width=WIDTH,
                    height=HEIGHT
                )
            )
        for i in range(SQUARE_BUBBLES):
            self.geometric.append(
                Square(
                    surface=self.surface,
                    surface_color=self.surface_color,
                    bubbles_color=BUBBLES_COLOR,
                    radius=BUBBLES_RADIUS,
                    width=WIDTH,
                    height=HEIGHT
                )
            )

    def random_trajectory(self):
        for geometric in self.geometric:
            if geometric.check_board_collision():
                geometric.board_collision()
            if geometric.check_collision():
                geometric.elastic_collision(self.take_the_nearest(geometric))
            
            geometric.show()

    # def circular_trajectory(self, TRAJETORY_RADIUS):
    #     self.geometric[0].move_circular()
    #     self.geometric[0].show()
            
    # def square_trajectory(self, TRAJETORY_RADIUS):
    #     self.geometric[0].move_square()
    #     self.geometric[0].show()

    def take_the_nearest(self, geometric):
        distance = []

        for nearest in self.geometric:
            if nearest != geometric:
                distance.append(math.sqrt(((geometric.x-nearest.x)**2)+((geometric.y-nearest.y)**2)))

        return self.geometric[distance.index(min(distance))]

    def show(self):
        self.surface.fill(self.surface_color)
        self.route_random()

        pygame.display.flip()
        pygame.time.Clock().tick(self.fps)

    # def show(self, TRAJETORY_TYPE='random', TRAJETORY_RADIUS=0):
    #     self.surface.fill(self.surface_color)

    #     if TRAJETORY_TYPE == 'random':
    #         self.route_random()
    #     elif TRAJETORY_TYPE == 'circular':
    #         self.route_circular(TRAJETORY_RADIUS)
    #     elif TRAJETORY_TYPE == 'square':
    #         self.route_square(TRAJETORY_RADIUS)

    #     pygame.display.flip()
    #     pygame.time.Clock().tick(self.fps)
    
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


