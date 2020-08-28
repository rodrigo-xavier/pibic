import pygame
from pygame.locals import *

import random, math, sys
from bubbles import Bubbles


# Configurations
SURFACE_COLOR = (0,0,0)
BUBBLES_COLOR = (255,255,255)
WIDTH, HEIGHT = 250, 250
FPS = 30
NUMBER_OF_DATA = 200
BUBBLES_RADIUS = 15
CIRCLE_BUBBLES = 10
SQUARE_BUBBLES = 0



def run(bubbles):
    for i in range(0, NUMBER_OF_DATA):
    # while True:
        bubbles.close()
        bubbles.show()
        # bubbles.save(i)
    # bubbles.img2npz()
    pygame.quit(); sys.exit()



# Init Game
pygame.init()
bubbles = Bubbles(SURFACE_COLOR, FPS, CIRCLE_BUBBLES, SQUARE_BUBBLES, BUBBLES_COLOR, BUBBLES_RADIUS, WIDTH, HEIGHT)
run(bubbles)