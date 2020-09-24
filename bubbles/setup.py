import pygame
from pygame.locals import *

import random, math, sys
from game import BubblesGame
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
NUMBER_OF_FRAMES = 120
BUBBLES_RADIUS = 2
CIRCLE_BUBBLES = 1
SQUARE_BUBBLES = 0

################# Trajectory #################

TRAJECTORY_RADIUS = 25
LAPS = 5

################# Select Trajectory #################

# TRAJECTORY = 'random'
TRAJECTORY = 'circular'
# TRAJECTORY = 'square'
INCREASE_RADIUS = True
SAVE = True
NPZ = True




####################################### GAMESPACE #######################################

# 120 == Frames per lap
def run(bubbles, loop_counter):
    data = AIData(IMG_PATH, NPZ_PATH, TRAJECTORY, WIDTH, HEIGHT, CIRCLE_BUBBLES, SQUARE_BUBBLES)

    if INCREASE_RADIUS and loop_counter == 1:
        data.reset_folder()
    elif not INCREASE_RADIUS:
        data.reset_folder()

    if TRAJECTORY != 'random':
        frames = LAPS * 120
    else:
        frames = NUMBER_OF_FRAMES

    for i in range(1, frames):
        bubbles.close()
        bubbles.show()

        file_name = str(loop_counter) + str(i)

        if SAVE and TRAJECTORY != 'random' and LAPS % 5 == 0:
            if i >= 1 and i <= 120:
                bubbles.save(file_name, IMG_PATH)
            if i >= 150 and i <= 270:
                bubbles.save(file_name, IMG_PATH)
            if i >= 300 and i <= 420:
                bubbles.save(file_name, IMG_PATH)
            if i >= 450 and i <= 570:
                bubbles.save(file_name, IMG_PATH)
        elif SAVE and TRAJECTORY == 'random':
            bubbles.save(i, IMG_PATH)
    
    if NPZ and not INCREASE_RADIUS:
        data.img2npz()
    elif NPZ and INCREASE_RADIUS and loop_counter == (TRAJECTORY_RADIUS-1):
        print("here too")
        data.img2npz()

def build_all():
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
    loop_counter = 0

    if INCREASE_RADIUS and TRAJECTORY != 'random':
        for i in range(1, TRAJECTORY_RADIUS):
            loop_counter+=1
            bubbles.bubbles[0].trajectory_radius = i
            run(bubbles, loop_counter)
    else:
        bubbles.bubbles[0].trajectory_radius = TRAJECTORY_RADIUS
        run(bubbles, loop_counter)

################# Init #################

pygame.init()
build_all()
pygame.quit(); sys.exit()


##########################################################################################