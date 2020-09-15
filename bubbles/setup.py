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
NUMBER_OF_DATA = 20000
BUBBLES_RADIUS = 25
CIRCLE_BUBBLES = 1
SQUARE_BUBBLES = 0

################# Trajectory #################

# TRAJECTORY = 'random'
TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
TRAJECTORY_RADIUS = 125




################# RUN #################

def run(bubbles):
    if TRAJECTORY != 'random':
        FRAMES_PER_VOLTA = bubbles.find_frames_per_turn()
        print(FRAMES_PER_VOLTA)

    for i in range(0, NUMBER_OF_DATA):
        # print(i)
        bubbles.close()
        bubbles.show()
        # bubbles.save(i)

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