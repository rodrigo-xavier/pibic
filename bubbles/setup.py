import pygame
from pygame.locals import *

import random, math, sys
from game import BubblesGame
from data import AIData
import copy


################# Database Location #################

IMG_PATH = "../../.database/pibic/pygame/img/"
NPZ_PATH = "../../.database/pibic/pygame/npz/"

################# Configurations #################

SURFACE_COLOR = (0,0,0)
BUBBLES_COLOR = (255,255,255)
WIDTH, HEIGHT = 500, 500
FPS = 0
NUMBER_OF_FRAMES = 12000
BUBBLES_RADIUS = 25
CIRCLE_BUBBLES = 6
SQUARE_BUBBLES = 0

################# Trajectory #################

TRAJECTORY_RADIUS = 125
FRAMES_PER_LAP_APPROXIMATELY = 120
NUMBER_OF_LAPS = 1

################# Select Trajectory #################

TRAJECTORY = 'random'
# TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
SAVE = False

################# RUN #################

def run(bubbles):
    data = AIData(IMG_PATH, NPZ_PATH)
    frames_per_lap = 0
    frames = NUMBER_OF_FRAMES

    if TRAJECTORY != 'random':
        frames_per_lap_actual = bubbles.find_frames_per_lap()
        frames_per_lap = int(frames_per_lap_actual / FRAMES_PER_LAP_APPROXIMATELY)
        frames = frames_per_lap_actual*NUMBER_OF_LAPS

    for i in range(1, frames):
        bubbles.close()
        bubbles.show()

        if SAVE and TRAJECTORY != 'random' and i%frames_per_lap == 0:
            bubbles.save(int(i/frames_per_lap), IMG_PATH)
        elif SAVE and TRAJECTORY == 'random':
            bubbles.save(i, IMG_PATH)

################# Game #################

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




################# Prepare Data for AI #################

# data = AIData(
#     IMG_PATH, 
#     NPZ_PATH,
# )
# data.img2npz()