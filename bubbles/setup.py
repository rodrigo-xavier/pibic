import pygame
from pygame.locals import *

import random, math, sys
from game import BubblesGame
from data import AIData


##################################### CONFIGURATIONS #####################################


################# Database Location #################

IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"

################# Workflow #################

SURFACE_COLOR = (0,0,0)
BUBBLES_COLOR = (255,255,255)
WIDTH, HEIGHT = 500, 500
FPS = 0
NUMBER_OF_FRAMES = 1000
BUBBLES_RADIUS = 25
CIRCLE_BUBBLES = 6
SQUARE_BUBBLES = 0

################# Trajectory #################

TRAJECTORY_RADIUS = 125
IMGS_PER_LAP_APPROXIMATELY = 120
LAPS = 1

################# Select Trajectory #################

TRAJECTORY = 'random'
# TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
SAVE = True
NPZ = True


####################################### GAMESPACE #######################################

# 120 == Frames per lap
def run(bubbles):
    data = AIData(IMG_PATH, NPZ_PATH, TRAJECTORY, WIDTH, HEIGHT)
    data.reset_folder()

    if TRAJECTORY != 'random':
        frames = LAPS * 120
    else:
        frames = NUMBER_OF_FRAMES

    for i in range(1, frames):
        bubbles.close()
        bubbles.show()

        if SAVE and TRAJECTORY != 'random' and i % 120 == 0:
            bubbles.save(int(i / 120), IMG_PATH)
        elif SAVE and TRAJECTORY == 'random':
            bubbles.save(i, IMG_PATH)
    
    if NPZ:
        data.img2npz()

################# Init #################

pygame.init()

bubbles = BubblesGame(
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

pygame.quit(); sys.exit()


##########################################################################################