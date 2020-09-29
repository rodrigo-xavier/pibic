import pygame
from pygame.locals import *

import random, math, sys
from data import AIData


##################################### CONFIGURATIONS #####################################


################# Database Location #################

IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"
# IMG_PATH = "../../database/img/"
# NPZ_PATH = "../../database/npz/"

################# Workflow #################

SURFACE_COLOR = (0,0,0)
BUBBLES_COLOR = (255,255,255)
WIDTH, HEIGHT = 50, 50
FPS = 0
NUMBER_OF_FRAMES = 120000
BUBBLES_RADIUS = 3
CIRCLE_BUBBLES = 0
SQUARE_BUBBLES = 1

################# Trajectory #################

TRAJECTORY_RADIUS = 25
LAPS = 5

################# Select Trajectory #################

# TRAJECTORY = 'random'
TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
INCREASE_RADIUS = True
SAVE = False
NPZ = False




####################################### GAMESPACE #######################################


def init_game():
    data = AIData(
        IMG_PATH=IMG_PATH,
        NPZ_PATH=NPZ_PATH,
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
        INCREASE_RADIUS=INCREASE_RADIUS,
    )
    data.reset_folder()

    # 120 == Frames per lap
    if TRAJECTORY != 'random' and LAPS % 5 == 0:
        data.play_game(LAPS * 120, SAVE)
    else:
        data.play_game(NUMBER_OF_FRAMES, SAVE)

    if NPZ:
        data.img2npz()   
    

################# Init #################

pygame.init()
init_game()
pygame.quit(); sys.exit()


##########################################################################################