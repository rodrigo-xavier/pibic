import random, math
import pygame
from pygame.locals import *
import numpy as np

class Square:
    def __init__(self, surface=None, BUBBLES_COLOR=(0,0,0), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50):
        
        self.surface = surface
        self.width = WIDTH
        self.height = HEIGHT
        self.color = BUBBLES_COLOR
        
        self.x = random.randint(BUBBLES_RADIUS, self.width-BUBBLES_RADIUS)
        self.y = random.randint(BUBBLES_RADIUS, self.height-BUBBLES_RADIUS)
        self.z = math.sqrt(2*(BUBBLES_RADIUS**2)) # Largura
        self.w = math.sqrt(2*(BUBBLES_RADIUS**2)) # Comprimento

        self.m = random.random()
        self.v = np.array([random.random(), random.random()])

        # self.radius = int((math.sqrt(2*(self.side**2))) / 2) # Describes the circumference that cover the square
    
    def board_collision(self):
        if self.x <= 0 or self.x >= (self.width - self.z):
            self.v[0] *= -1
        if self.y <= 0 or self.y >= (self.height - self.w):
            self.v[1] *= -1
        
        self.move()
    
    def check_collision(self):
        for x in range(int(self.x), int(self.x + self.z)):
            y = self.x + self.z

            if not (x <= 0 or x >= self.width or y <= 0 or y >= self.height):
                color1 = self.surface.get_at((int(x), int(self.y)))
                color2 = self.surface.get_at((int(x), int(y)))

                if (color1 == self.color) or (color2 == self.color):
                    return True

        for y in range(int(self.y), int(self.y + self.w)):
            x = self.x + self.z

            if not (x <= 0 or x >= self.width or y <= 0 or y >= self.height):
                color1 = self.surface.get_at((int(self.x), int(y)))
                color2 = self.surface.get_at((int(x), int(y)))

                if (color1 == self.color) or (color2 == self.color):
                    return True
    
        return False
    
    def elastic_collision(self, other_circle):
        v1i = self.v
        v2i = other_circle.v

        m1 = self.m
        m2 = other_circle.m

        self.v = ((v1i*(m1-m2)) + (2*v2i*m2)) / (m1 + m2)
        other_circle.v = ((v2i*(m2-m1)) + (2*v1i*m1)) / (m1 + m2)

        self.move()
    
    def move(self):
        self.x += self.v[0]
        self.y += self.v[1]

    def show(self):
        pygame.draw.rect(self.surface, self.color,(int(self.x),int(self.y),int(self.z),int(self.w)))