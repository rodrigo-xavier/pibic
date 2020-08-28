import random, math
import pygame
from pygame.locals import *

class Square:
    def __init__(self, surface=None, BUBBLES_COLOR=(0,0,0), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50):
        self.surface = surface
        self.width = WIDTH
        self.height = HEIGHT
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS
        self.x = random.randint(self.radius+1, self.width-self.radius-1)
        self.y = random.randint(self.radius+1, self.height-self.radius-1)
        self.z = math.sqrt(2*(self.radius**2))
        self.w = math.sqrt(2*(self.radius**2))
        self.speedx = random.random()
        self.speedy = random.random()

        # self.radius = int((math.sqrt(2*(self.side**2))) / 2) # Describes the circumference that cover the square
    
    def board_collision(self):
        if self.x < self.radius or self.x > self.width-self.radius:
            self.speedx *= -1
        if self.y < self.radius or self.y > self.height-self.radius:
            self.speedy *= -1
    
    def square_collision(self, other_square):
        if math.sqrt(((self.x-other_square.x)**2)+((self.y-other_square.y)**2)) <= (self.radius+other_square.radius):
            self.collision(other_square)
    
    # def collision(self):
    #     self.speedx *= -1
    #     self.speedy *= -1
    
    def collision(self, other_square):
        speed = math.sqrt((self.speedx**2)+(self.speedy**2))
        diffx = -(self.x-other_square.x)
        diffy = -(self.y-other_square.y)
        if diffx > 0:
            if diffy > 0:
                angle = math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
            elif diffy < 0:
                angle = math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
        elif diffx < 0:
            if diffy > 0:
                angle = 180 + math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
            elif diffy < 0:
                angle = -180 + math.degrees(math.atan(diffy/diffx))
                speedx = -speed*math.cos(math.radians(angle))
                speedy = -speed*math.sin(math.radians(angle))
        elif diffx == 0:
            if diffy > 0:
                angle = -90
            else:
                angle = 90
            speedx = speed*math.cos(math.radians(angle))
            speedy = speed*math.sin(math.radians(angle))
        elif diffy == 0:
            if diffx < 0:
                angle = 0
            else:
                angle = 180
            speedx = speed*math.cos(math.radians(angle))
            speedy = speed*math.sin(math.radians(angle))
        self.speedx = speedx
        self.speedy = speedy
    
    def move(self):
        self.x += self.speedx
        self.y += self.speedy

    def show(self):
        pygame.draw.rect(self.surface, self.color,(int(self.x),int(self.y),int(self.z),int(self.w)))