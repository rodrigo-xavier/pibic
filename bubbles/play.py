import pygame
from pygame.locals import *

import sys
import cv2
import numpy as np
import os
from circle import Circle
from square import Square
import math, random


IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"


class Play:
    bubbles = []
    tensor = []

    def __init__(self, SURFACE_COLOR=(0,0,0), FPS=60, CIRCLE_BUBBLES=1, SQUARE_BUBBLES=1, BUBBLES_COLOR=(255,255,255), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50):
        
        self.surface = pygame.display.set_mode((WIDTH,HEIGHT))
        self.surface_color = SURFACE_COLOR
        self.fps = FPS

        for i in range(CIRCLE_BUBBLES):
            x, y = self.build_far(BUBBLES_RADIUS, WIDTH, HEIGHT)
            self.bubbles.append(
                Circle(
                    surface=self.surface,
                    surface_color=self.surface_color,
                    bubbles_color=BUBBLES_COLOR,
                    bubbles_radius=BUBBLES_RADIUS,
                    width=WIDTH,
                    height=HEIGHT,
                    x=x,
                    y=y,
                )
            )
        for i in range(SQUARE_BUBBLES):
            x, y = self.build_far(BUBBLES_RADIUS, WIDTH, HEIGHT)
            self.bubbles.append(
                Square(
                    surface=self.surface,
                    surface_color=self.surface_color,
                    bubbles_color=BUBBLES_COLOR,
                    bubbles_radius=BUBBLES_RADIUS,
                    width=WIDTH,
                    height=HEIGHT,
                    x=x,
                    y=y,
                )
            )

    def build_far(self, radius, width, height):
        OFFSET = 5
        far = False
        distance = []

        if not self.bubbles:
            x = random.randint(radius+OFFSET, width-radius-OFFSET)
            y = random.randint(radius+OFFSET, height-radius-OFFSET)
            return x, y

        while not far:
            distance = []
            x = random.randint(radius+OFFSET, width-radius-OFFSET)
            y = random.randint(radius+OFFSET, height-radius-OFFSET)

            for bubble in self.bubbles:
                distance.append(math.sqrt(((x-bubble.x)**2) + ((y-bubble.y)**2)))

            if min(distance) > (2*radius + OFFSET):
                far = True
        
        return x, y

    def random_trajectory(self):
        for bubble in self.bubbles:
            if bubble.check_board_collision():
                bubble.board_collision()
            elif self.have_collision(bubble):
                bubble.elastic_collision(self.take_the_nearest(bubble))

            bubble.move()
            bubble.show()

    # def circular_trajectory(self, TRAJETORY_RADIUS):
    #     self.bubbles[0].move_circular()
    #     self.bubbles[0].show()
            
    # def square_trajectory(self, TRAJETORY_RADIUS):
    #     self.bubbles[0].move_square()
    #     self.bubbles[0].show()

    def have_collision(self, bubble):
        OFFSET = 2

        newx = bubble.x + bubble.v[0]
        newy = bubble.y + bubble.v[1]

        for nearest in self.bubbles:
            if math.sqrt(((newx-nearest.x)**2)+((newy-nearest.y)**2)) <= (bubble.radius + nearest.radius + OFFSET):
                return bubble.check_collision()
        return False

    def take_the_nearest(self, bubble):
        distance = []

        for nearest in self.bubbles:
            if nearest != bubble:
                distance.append(math.sqrt(((bubble.x-nearest.x)**2)+((bubble.y-nearest.y)**2)))

        return self.bubbles[distance.index(min(distance))]

    def show(self):
        self.surface.fill(self.surface_color)
        self.random_trajectory()

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


