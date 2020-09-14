import random, math
import pygame
from pygame.locals import *
import numpy as np
from abc import ABC, abstractmethod


# self.x = ponto da coordenada horizontal da superficie
# self.y = ponto da coordenada vertical da superficie
# self.w = Largura
# self.z = Comprimento
# self.m = Massa da esfera
# self.v = Tupla com velocidades na componente i e j
# self.width = Largura da janela
# self.height = Comprimento da janela
# self.radius = Nesse caso, o raio sera o raio do circulo que engloba toda a bolha

class Bubbles(ABC):
    def __init__(self, *args, **kwargs):
        
        self.surface = kwargs['surface']
        self.surface_color = kwargs['surface_color']
        self.bubbles_color = kwargs['bubbles_color']
        self.radius = kwargs['bubbles_radius'] # Nesse caso, o raio sera o raio do circulo que engloba toda a bolha
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.x = kwargs['x']
        self.y = kwargs['y']

        self.w = math.sqrt(2*(kwargs['bubbles_radius']**2)) # Largura
        self.z = math.sqrt(2*(kwargs['bubbles_radius']**2)) # Comprimento

        self.m = random.random()
        self.v = np.array([random.random(), random.random()])

        self.board_collision_x = False
        self.board_collision_y = False

    # https://pt.wikipedia.org/wiki/Colis%C3%A3o_el%C3%A1stica#:~:text=Em%20f%C3%ADsica%2C%20uma%20colis%C3%A3o%20el%C3%A1stica,deforma%C3%A7%C3%B5es%20permanentes%20durante%20o%20impacto.
    # vf = (v1i*(m1-m2) + 2*v2i*m2) / (m1 + m2)
    def elastic_collision(self, bubble):
        v1i = self.v
        v2i = bubble.v

        m1 = self.m
        m2 = bubble.m

        self.v = ((v1i*(m1-m2)) + (2*v2i*m2)) / (m1 + m2)
        bubble.v = ((v2i*(m2-m1)) + (2*v1i*m1)) / (m1 + m2)
    
    def board_collision(self):
        if self.board_collision_x:
            self.board_collision_x = False
            self.v[0] *= -1
        if self.board_collision_y:
            self.board_collision_y = False
            self.v[1] *= -1
    
    def move(self):
        self.x += self.v[0]
        self.y += self.v[1]
    
    def move_circular(self):
        if self.ang_idx == self.n_angles:
            self.ang_idx = 0
            self.n_loops = self.n_loops + 1

        x_center = self.tragetory_radius * math.cos(self.angles[self.ang_idx])
        y_center = self.tragetory_radius * math.sin(self.angles[self.ang_idx])

        self.ang_idx = self.ang_idx + 1

        self.x = self.CIRCULAR_CENTER[0] + x_center
        self.y = self.CIRCULAR_CENTER[1] + y_center
    
    def move_square(self):
        if self.x == (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and (self.y < self.CIRCULAR_CENTER[0] + self.tragetory_radius):
            self.y = self.y + 1
        elif self.x > (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y == (self.CIRCULAR_CENTER[0] + self.tragetory_radius):
            self.x = self.x - 1
        elif self.x == (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y > (self.CIRCULAR_CENTER[0] - self.tragetory_radius):
            self.y = self.y - 1
        elif self.x < (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and self.y == (self.CIRCULAR_CENTER[0] - self.tragetory_radius):
            self.x = self.x + 1
    

    def show_pixel(self, x, y):
        pygame.draw.circle(self.surface, (255,0,0), (int(x),int(y)), 1)