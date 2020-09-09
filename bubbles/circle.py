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
    def __init__(self, surface=None, BUBBLES_COLOR=(0,0,0), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50, MOVEMENT_SHAPE='circular', TRAGETORY_RADIUS=12):
        
        self.surface = surface
        self.width = WIDTH
        self.height = HEIGHT
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS

        self.CIRCULAR_CENTER = (int(WIDTH/2), int(HEIGHT/2))
        # self.x = random.randint(self.radius, self.width-self.radius)
        # self.y = random.randint(self.radius, self.height-self.radius)
        self.x = self.CIRCULAR_CENTER[0] +  TRAGETORY_RADIUS
        self.y = self.CIRCULAR_CENTER[1]
        self.m = random.random()
        self.v = np.array([random.random(), random.random()])
    
        self.ang_idx = 0

        self.angles = np.linspace(0, 2*np.pi, 120, endpoint=True)
        self.n_loops = 0
        self.movement_shape = MOVEMENT_SHAPE
        self.tragetory_radius = TRAGETORY_RADIUS
        self.n_angles = len(self.angles) - 1

    def board_collision(self):
        if self.x <= (self.radius) or self.x >= (self.width - self.radius):
            self.v[0] *= -1
        if self.y <= (self.radius) or self.y >= (self.height - self.radius):
            self.v[1] *= -1
        
        self.move()

    def check_collision(self):
        r2 = self.radius * self.radius

        for x1 in range(int(self.x - self.radius), int(self.x + self.radius)):
            y1 = int(math.sqrt(math.fabs(r2 - math.pow(x1 - self.x, 2))) + self.y + 1)
            y2 = int(self.y - (y1 - self.y))

            if not (x1 <= 0 or x1 >= self.width or y1 <= 0 or y1 >= self.height or y2 <= 0 or y2 >= self.height):
                color_y1 = self.surface.get_at((x1, y1))
                color_y2 = self.surface.get_at((x1, y2))

                if (color_y1 == self.color) or (color_y2 == self.color):
                    return True

        for y1 in range(int(self.y - self.radius), int(self.y + self.radius)):
            x1 = int(math.sqrt(math.fabs(r2 - math.pow(y1 - self.y, 2))) + self.x + 1)
            x2 = int(self.x - (x1 - self.x))

            if not (y1 <= 0 or y1 >= self.height or x1 <= 0 or x1 >= self.width or x2 <= 0 or x2 >= self.width):
                color_x1 = self.surface.get_at((x1, y1))
                color_x2 = self.surface.get_at((x2, y1))

                if (color_x1 == self.color) or (color_x2 == self.color):
                    return True
    
        return False

    # https://pt.wikipedia.org/wiki/Colis%C3%A3o_el%C3%A1stica#:~:text=Em%20f%C3%ADsica%2C%20uma%20colis%C3%A3o%20el%C3%A1stica,deforma%C3%A7%C3%B5es%20permanentes%20durante%20o%20impacto.
    # vf = (v1i*(m1-m2) + 2*v2i*m2) / (m1 + m2)
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
            if self.x => (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and (self.y < self.CIRCULAR_CENTER[0] + self.tragetory_radius):
                self.y = self.y + 1
            elif self.x > (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y > (self.CIRCULAR_CENTER[0] + self.tragetory_radius):
                self.x = self.x - 1
            elif self.x == (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y > (self.CIRCULAR_CENTER[0] - self.tragetory_radius):
                self.y = self.y - 1
            elif self.x < (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and self.y < (self.CIRCULAR_CENTER[0] - self.tragetory_radius):
                self.x = self.x + 1
            '''
            
            if self.x >= (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and (self.y > self.CIRCULAR_CENTER[1] - self.tragetory_radius):
                self.y = self.y - tragetory_step
            elif self.x > (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y <= (self.CIRCULAR_CENTER[1] - self.tragetory_radius):
                self.x = self.x - tragetory_step
            elif self.x <= (self.CIRCULAR_CENTER[0] - self.tragetory_radius) and self.y < (self.CIRCULAR_CENTER[1] + self.tragetory_radius):
                self.y = self.y + tragetory_step
            elif self.x < (self.CIRCULAR_CENTER[0] + self.tragetory_radius) and self.y >= (self.CIRCULAR_CENTER[1] + self.tragetory_radius):
                self.x = self.x + tragetory_step
    
    def show(self):
        pygame.draw.circle(self.surface, self.color, (int(self.x),int(self.y)), self.radius)