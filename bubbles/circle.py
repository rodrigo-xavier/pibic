import random, math
import pygame
from pygame.locals import *
import numpy as np
from abstract import Bubbles

# self.x = horizontal da superficie
# self.y = vertical da superficie
# self.m = Massa da esfera
# self.v = Tupla com velocidades na componente i e j
# self.width = Largura da janela
# self.height = Comprimento da janela

class Circle(Bubbles):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        # self.CIRCULAR_CENTER = (int(WIDTH/2), int(HEIGHT/2))
        # self.x = self.CIRCULAR_CENTER[0] +  int(self.CIRCULAR_CENTER[0]/2)
        # self.y = self.CIRCULAR_CENTER[1]
    
        # self.ang_idx = 0

        # self.angles = np.linspace(0, 2*np.pi, TRAGETORY_RADIUS * 8)
        # self.n_loops = 0
        # self.movement_shape = MOVEMENT_SHAPE
        # self.tragetory_radius = TRAGETORY_RADIUS
        # self.n_angles = TRAGETORY_RADIUS * 8

    def check_collision(self):
        for theta in range(0, 360):
            x = int ((self.radius * math.cos(theta)) + self.x)
            y = int ((self.radius * math.sin(theta)) + self.y)
        
            color = self.surface.get_at((x, y))

            if color != self.surface_color:
                return True

        return False

    def check_board_collision(self):
        OFFSET = 2

        if self.x <= (self.radius + OFFSET) or self.x >= (self.width - self.radius - OFFSET):
            self.board_collision_x = True
            return True
        if self.y <= (self.radius + OFFSET) or self.y >= (self.height - self.radius - OFFSET):
            self.board_collision_y = True
            return True
        return False

    def show(self):
        pygame.draw.circle(self.surface, self.bubbles_color, (int(self.x),int(self.y)), self.radius)