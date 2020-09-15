import pygame
from pygame.locals import *

import random, math, sys
from play import Play


# Database Location
IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"

# Configurations
SURFACE_COLOR = (0,0,0)
BUBBLES_COLOR = (255,255,255)
WIDTH, HEIGHT = 500, 500
FPS = 100
NUMBER_OF_DATA = 200
BUBBLES_RADIUS = 25
CIRCLE_BUBBLES = 1
SQUARE_BUBBLES = 0

# TRAJECTORY = 'random'
TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
TRAJECTORY_RADIUS = 125



def run(bubbles):
    # for i in range(0, NUMBER_OF_DATA):
    while True:
        bubbles.close()
        bubbles.show()
        # bubbles.save(i)
    # bubbles.img2npz()
    pygame.quit(); sys.exit()



# Init Game
pygame.init()

bubbles = Play(
    IMG_PATH, 
    NPZ_PATH,
    SURFACE_COLOR,
    FPS,
    CIRCLE_BUBBLES,
    SQUARE_BUBBLES,
    BUBBLES_COLOR,
    BUBBLES_RADIUS,
    WIDTH,
    HEIGHT,
    TRAJECTORY,
    TRAJECTORY_RADIUS,
)

run(bubbles)