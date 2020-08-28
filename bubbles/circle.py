import random, math
import pygame
from pygame.locals import *

class Circle:
    def __init__(self, surface=None, bubbles_color=(0,0,0), bubbles_radius=1, width=50, height=50):
        self.surface = surface
        self.width = width
        self.height = height
        self.color = bubbles_color
        self.radius = bubbles_radius
        self.x = random.randint(self.radius+1, self.width-self.radius-1)
        self.y = random.randint(self.radius+1, self.height-self.radius-1)
        self.speedx = random.random()
        self.speedy = random.random()
    
    def board_collision(self):
        if self.x < self.radius or self.x > self.width-self.radius:
            self.speedx *= -1
        if self.y < self.radius or self.y > self.height-self.radius:
            self.speedy *= -1
    
    def circle_collision(self, other_circle):
        if math.sqrt(((self.x-other_circle.x)**2)+((self.y-other_circle.y)**2)) <= (self.radius+other_circle.radius):
            self.collision(other_circle)
    
    # def collision(self):
    #     self.speedx *= -1
    #     self.speedy *= -1

    def collision(self, other_circle):
        speed = math.sqrt((self.speedx**2)+(self.speedy**2))
        diffx = -(self.x-other_circle.x)
        diffy = -(self.y-other_circle.y)
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
        pygame.draw.circle(self.surface, self.color, (int(self.x),int(self.height-self.y)), self.radius)