import pygame
from pygame.locals import *

import random, math, sys
from bubbles import Bubbles


# Configurations
SURFACE_COLOR = (0,0,0)
BUBBLES_COLOR = (255,255,255)
WIDTH, HEIGHT = 50, 50
FPS = 100
NUMBER_OF_DATA = 500
BUBBLES_RADIUS = 30
CIRCLE_BUBBLES = 0
SQUARE_BUBBLES = 1
# MOVEMENT_SHAPE = 'circular'
MOVEMENT_SHAPE = 'square'
TRAGETORY_RADIUS = 12
VOLTA = 10

def run(bubbles):
    n = NUMBER_OF_DATA*VOLTA
    
    for i in range(0, n):
    # while True:
        bubbles.close()
        bubbles.show()
        bubbles.save(i)
    bubbles.img2npz_trajetory()
    pygame.quit(); sys.exit()


# Init Game
pygame.init()
bubbles = Bubbles(SURFACE_COLOR, FPS, CIRCLE_BUBBLES, SQUARE_BUBBLES, BUBBLES_COLOR, BUBBLES_RADIUS, WIDTH, HEIGHT, MOVEMENT_SHAPE, TRAGETORY_RADIUS)
run(bubbles)