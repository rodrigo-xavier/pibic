import pygame
from pygame.locals import *

import sys
import cv2
import numpy as np
import os
from circle import Circle
from square import Square


IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"


class Bubbles:
    circles = []
    squares = []
    tensor = []

    def __init__(self, SURFACE_COLOR=(0,0,0), FPS=60, CIRCLE_BUBBLES=1, SQUARE_BUBBLES=1, BUBBLES_COLOR=(255,255,255), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50):
        
        self.surface = pygame.display.set_mode((WIDTH,HEIGHT))
        self.surface_color = SURFACE_COLOR
        self.fps = FPS
        self.bubbles_color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS
        self.width = WIDTH
        self.height = HEIGHT

        for i in range(CIRCLE_BUBBLES):
            self.circles.append(Circle(self.surface, self.bubbles_color, self.radius, self.width, self.height))
        for j in range(SQUARE_BUBBLES):
            self.squares.append(Square(self.surface, self.bubbles_color, self.radius, self.width, self.height))
    
    def move(self):
        for circle in self.circles:
            circle.check_board_collision()

            for other_circle in self.circles:
                if circle != other_circle:
                    circle.check_circle_collision(other_circle)
            
            if self.squares is not None:
                for square in self.squares:
                    circle.check_circle_collision(square)
            
            circle.move()
            circle.show()
        
        for square in self.squares:
            square.check_board_collision()

            for other_square in self.squares:
                if square != other_square:
                    square.check_square_collision(other_square)
            
            if self.circles is not None:
                for circle in self.circles:
                    square.check_square_collision(circle)
            
            square.move()
            square.show()

    # def move_just_square(self):
    #     for square in self.squares:
    #         square.board_collision()

    #         for other_square in self.squares:
    #             if square != other_square:
    #                 square.square_collision(other_square)
            
    #         square.move()
    #         square.show()

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