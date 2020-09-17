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
CIRCLE_BUBBLES = 1
SQUARE_BUBBLES = 0

################# Trajectory #################

TRAJECTORY_RADIUS = 125
IMGS_PER_LAP_APPROXIMATELY = 120
NUMBER_OF_LAPS = 1

################# Select Trajectory #################

# TRAJECTORY = 'random'
TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
SAVE = True
NPZ = True


####################################### GAMESPACE #######################################


def run(bubbles):
    data = AIData(IMG_PATH, NPZ_PATH, TRAJECTORY, WIDTH, HEIGHT, CIRCLE_BUBBLES, SQUARE_BUBBLES)
    data.reset_folder()

    imgs_per_lap = 0
    frames = NUMBER_OF_FRAMES

    if TRAJECTORY != 'random':
        old_frames_per_lap = bubbles.find_frames_per_lap()
        new_frames_per_lap = int(old_frames_per_lap / IMGS_PER_LAP_APPROXIMATELY)
        imgs_per_lap = int(old_frames_per_lap/new_frames_per_lap) - 1

        frames = old_frames_per_lap*NUMBER_OF_LAPS

    for i in range(1, frames):
        bubbles.close()
        bubbles.show()

        if SAVE and TRAJECTORY != 'random' and i%new_frames_per_lap == 0:
            bubbles.save(int(i/new_frames_per_lap), IMG_PATH)
        elif SAVE and TRAJECTORY == 'random':
            bubbles.save(i, IMG_PATH)
    
    if NPZ:
        data.img2npz(imgs_per_lap)

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