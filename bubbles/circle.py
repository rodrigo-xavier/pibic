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
        newx = self.x + self.v[0]
        newy = self.y + self.v[1]

        angle = math.atan2(newy-self.y, newx-self.x)

        x = int ((self.radius * math.cos(angle)) + newx)
        y = int ((self.radius * math.sin(angle)) + newy)

        color = self.surface.get_at((x, y))

        if color != self.surface_color:
            return True
        return False
    
    def board_collision(self):
        OFFSET = 5
        
        newx = self.x + self.v[0]
        newy = self.y + self.v[1]

        if newx <= (self.radius + OFFSET) or newx >= (self.width - self.radius - OFFSET):
            self.v[0] *= -1
            self.x += self.v[0]
        if newy <= (self.radius + OFFSET) or newy >= (self.height - self.radius - OFFSET):
            self.v[1] *= -1
            self.y += self.v[1]
        

    def move(self):
        self.board_collision()
        if self.check_collision():
            self.elastic_collision(self.take_the_nearest(self.list_of_bubbles[self.bubble_index]))
        
        self.x += self.v[0]
        self.y += self.v[1]

    def show(self):
        pygame.draw.circle(self.surface, self.bubbles_color, (int(self.x),int(self.y)), self.radius)