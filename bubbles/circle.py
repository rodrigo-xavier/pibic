import random, math
import pygame
from pygame.locals import *
import numpy as np


# self.x = horizontal da superficie
# self.y = vertical da superficie
# self.m = Massa da esfera
# self.v = Tupla com velocidades na componente i e j
# self.width = Largura da janela
# self.height = Comprimento da janela

class Circle:
    def __init__(self, surface=None, BUBBLES_COLOR=(0,0,0), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50):
        
        self.surface = surface
        self.width = WIDTH
        self.height = HEIGHT
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS

        self.x = random.randint(self.radius, self.width-self.radius)
        self.y = random.randint(self.radius, self.height-self.radius)
        self.m = random.random()
        self.v = np.array([random.random(), random.random()])
    
    def check_board_collision(self):
        if self.x < self.radius or self.x > self.width-self.radius:
            self.v[0] *= -1
        if self.y < self.radius or self.y > self.height-self.radius:
            self.v[1] *= -1
    
    def check_circle_collision(self, other_circle):
        if math.sqrt(((self.x-other_circle.x)**2)+((self.y-other_circle.y)**2)) <= (self.radius+other_circle.radius):
            self.elastic_collision(other_circle)

    # https://pt.wikipedia.org/wiki/Colis%C3%A3o_el%C3%A1stica#:~:text=Em%20f%C3%ADsica%2C%20uma%20colis%C3%A3o%20el%C3%A1stica,deforma%C3%A7%C3%B5es%20permanentes%20durante%20o%20impacto.
    # vf = (v1i*(m1-m2) + 2*v2i*m2) / (m1 + m2)
    def elastic_collision(self, other_circle):
        v1i = self.v
        v2i = other_circle.v

        m1 = self.m
        m2 = other_circle.m

        self.v = ((v1i*(m1-m2)) + (2*v2i*m2)) / (m1 + m2)
        other_circle.v = ((v2i*(m2-m1)) + (2*v1i*m1)) / (m1 + m2)
    
    def move(self):
        self.x += self.v[0]
        self.y += self.v[1]
    
    def show(self):
        pygame.draw.circle(self.surface, self.color, (int(self.x),int(self.y)), self.radius)