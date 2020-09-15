import random, math
import pygame
from pygame.locals import *
import numpy as np
from abstract import Bubbles

class Square(Bubbles):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        # self.surface = surface
        # self.width = WIDTH
        # self.height = HEIGHT
        # self.color = BUBBLES_COLOR
        
        # # self.x = random.randint(BUBBLES_RADIUS, self.width-BUBBLES_RADIUS)
        # # self.y = random.randint(BUBBLES_RADIUS, self.height-BUBBLES_RADIUS)
        
        # self.CIRCULAR_CENTER = (int(WIDTH/2), int(HEIGHT/2))
        # self.x = self.CIRCULAR_CENTER[0] +  int(self.CIRCULAR_CENTER[0]/2)
        # self.y = self.CIRCULAR_CENTER[1]

        # self.z = math.sqrt(2*(BUBBLES_RADIUS**2)) # Largura
        # self.w = math.sqrt(2*(BUBBLES_RADIUS**2)) # Comprimento

        # self.m = random.random()
        # self.v = np.array([random.random(), random.random()])



        # self.ang_idx = 0

        # self.angles = np.linspace(0, 2*np.pi, TRAGETORY_RADIUS * 8)
        # self.n_loops = 0
        # self.movement_shape = MOVEMENT_SHAPE
        # self.tragetory_radius = TRAGETORY_RADIUS
        # self.n_angles = TRAGETORY_RADIUS * 8

        # self.radius = int((math.sqrt(2*(self.side**2))) / 2) # Describes the circumference that cover the square
    
    def check_collision(self):
        for x in range(int(self.x - (self.z/2)), int(self.x + (self.z/2))):
            y = self.x + self.z

            if not (x <= 0 or x >= self.width or y <= 0 or y >= self.height):
                color1 = self.surface.get_at((int(x), int(self.y)))
                color2 = self.surface.get_at((int(x), int(y)))

                if (color1 != self.surface_color) or (color2 != self.surface_color):
                    return True

        for y in range(int(self.y - (self.w/2)), int(self.y + (self.w/2))):
            x = self.x + self.z

            if not (x <= 0 or x >= self.width or y <= 0 or y >= self.height):
                color1 = self.surface.get_at((int(self.x), int(y)))
                color2 = self.surface.get_at((int(x), int(y)))

                if (color1 != self.surface_color) or (color2 != self.surface_color):
                    return True
    
        return False
    
    def check_board_collision(self):
        OFFSET = 5
        
        newx = self.x + self.v[0] - (self.z/2)
        newy = self.y + self.v[1] - (self.w/2)

        if newx <= OFFSET or newx >= (self.width - self.w - OFFSET):
            self.board_collision_x = True
            return True
        if newy <= OFFSET or newy >= (self.height - self.w - OFFSET):
            self.board_collision_y = True
            return True
        return False

    def show(self):
        # Assume-se que x e y sao coordenadas do centro do quadrado. Porem, para desenhar o quadrado, 
        # as coordenadas que precisamos sao as coordenadas do vertice superior esquerdo, por isso
        # eh necessario ajustar as coordenadas sempre que for preciso

        x = self.x - self.z/2
        y = self.y - self.w/2
        pygame.draw.rect(self.surface, self.bubbles_color,(int(x),int(y),int(self.z),int(self.w)))