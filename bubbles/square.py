import random, math
import pygame
from pygame.locals import *
import numpy as np

class Square:
    def __init__(self, surface=None, BUBBLES_COLOR=(0,0,0), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50, MOVEMENT_SHAPE='circular', TRAGETORY_RADIUS=12):
        
        self.surface = surface
        self.width = WIDTH
        self.height = HEIGHT
        self.color = BUBBLES_COLOR
        
        # self.x = random.randint(BUBBLES_RADIUS, self.width-BUBBLES_RADIUS)
        # self.y = random.randint(BUBBLES_RADIUS, self.height-BUBBLES_RADIUS)
        self.CIRCULAR_CENTER = (int(WIDTH/2), int(HEIGHT/2))
        self.x = self.CIRCULAR_CENTER[0] + TRAGETORY_RADIUS
        self.y = self.CIRCULAR_CENTER[1]

        self.z = math.sqrt(2*(BUBBLES_RADIUS**2)) # Largura
        self.w = math.sqrt(2*(BUBBLES_RADIUS**2)) # Comprimento

        self.m = random.random()
        self.v = np.array([random.random(), random.random()])

        self.ang_idx = 0

        self.angles = np.linspace(0, 2*np.pi, TRAGETORY_RADIUS * 8)
        self.n_loops = 0
        self.movement_shape = MOVEMENT_SHAPE
        self.tragetory_radius = TRAGETORY_RADIUS
        self.n_angles = TRAGETORY_RADIUS * 8

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
    
    def elastic_collision(self, geometric):
        v1i = self.v
        v2i = geometric.v

        m1 = self.m
        m2 = geometric.m

        self.v = ((v1i*(m1-m2)) + (2*v2i*m2)) / (m1 + m2)
        geometric.v = ((v2i*(m2-m1)) + (2*v1i*m1)) / (m1 + m2)

        self.move()
    
    def move(self):
        # self.x += self.v[0]
        # self.y += self.v[1]

        if self.movement_shape == 'circular':

            if self.ang_idx == self.n_angles:
                self.ang_idx = 0
                self.n_loops = self.n_loops + 1

            x_center = self.tragetory_radius * math.cos(self.angles[self.ang_idx])
            y_center = - self.tragetory_radius * math.sin(self.angles[self.ang_idx])

            self.ang_idx = self.ang_idx + 1

            self.x = self.CIRCULAR_CENTER[0] + x_center
            self.y = self.CIRCULAR_CENTER[1] + y_center

        elif self.movement_shape == 'square':

            # 120/8 = 15
            tragetory_step = self.tragetory_radius/15

            '''
            if self.x == (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and (self.y < self.CIRCULAR_CENTER[0] + self.tragetory_radius):
                self.y = self.y + 1
            elif self.x > (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y == (self.CIRCULAR_CENTER[0] + self.tragetory_radius):
                self.x = self.x - 1
            elif self.x == (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y > (self.CIRCULAR_CENTER[0] - self.tragetory_radius):
                self.y = self.y - 1
            elif self.x < (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and self.y == (self.CIRCULAR_CENTER[0] - self.tragetory_radius):
                self.x = self.x + 1
            print(self.x, self.y)'''
            
            if self.x >= (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and (self.y > self.CIRCULAR_CENTER[1] - self.tragetory_radius):
                self.y = self.y - tragetory_step
            elif self.x > (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y <= (self.CIRCULAR_CENTER[1] - self.tragetory_radius):
                self.x = self.x - tragetory_step
            elif self.x <= (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y < (self.CIRCULAR_CENTER[1] + self.tragetory_radius):
                self.y = self.y + tragetory_step
            elif self.x < (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and self.y >= (self.CIRCULAR_CENTER[1] + self.tragetory_radius):
                self.x = self.x + tragetory_step

    def show(self):
        pygame.draw.rect(self.surface, self.color,(int(self.x),int(self.y),int(self.z),int(self.w)))