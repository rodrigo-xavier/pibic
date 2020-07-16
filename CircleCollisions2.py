import pygame
from pygame.locals import *
import random, math, sys


pygame.init()

width, height = 420, 420

Surface = pygame.display.set_mode((width,height))

Circles = []
class Circle:
    def __init__(self):
##        self.radius = int(random.random()*50) + 1
        self.radius = 20
        self.x = random.randint(self.radius, width-self.radius)
        self.y = random.randint(self.radius, height-self.radius)
        self.speedx = 0.5*(random.random()+1.0)
        self.speedy = 0.5*(random.random()+1.0)
##        self.mass = math.sqrt(self.radius)

Squares = []
class Square:
    def __init__(self):
        self.side = random.randint(1,50)
        self.x = random.randint(1, width)
        self.y = random.randint(1, height)
        self.z = self.side
        self.w = self.side


for x in range(20):
    Circles.append(Circle())
    Squares.append(Square())

def CircleCollide(C1,C2):
    C1Speed = math.sqrt((C1.speedx**2)+(C1.speedy**2))
    XDiff = -(C1.x-C2.x)
    YDiff = -(C1.y-C2.y)
    if XDiff > 0:
        if YDiff > 0:
            Angle = math.degrees(math.atan(YDiff/XDiff))
            XSpeed = -C1Speed*math.cos(math.radians(Angle))
            YSpeed = -C1Speed*math.sin(math.radians(Angle))
        elif YDiff < 0:
            Angle = math.degrees(math.atan(YDiff/XDiff))
            XSpeed = -C1Speed*math.cos(math.radians(Angle))
            YSpeed = -C1Speed*math.sin(math.radians(Angle))
    elif XDiff < 0:
        if YDiff > 0:
            Angle = 180 + math.degrees(math.atan(YDiff/XDiff))
            XSpeed = -C1Speed*math.cos(math.radians(Angle))
            YSpeed = -C1Speed*math.sin(math.radians(Angle))
        elif YDiff < 0:
            Angle = -180 + math.degrees(math.atan(YDiff/XDiff))
            XSpeed = -C1Speed*math.cos(math.radians(Angle))
            YSpeed = -C1Speed*math.sin(math.radians(Angle))
    elif XDiff == 0:
        if YDiff > 0:
            Angle = -90
        else:
            Angle = 90
        XSpeed = C1Speed*math.cos(math.radians(Angle))
        YSpeed = C1Speed*math.sin(math.radians(Angle))
    elif YDiff == 0:
        if XDiff < 0:
            Angle = 0
        else:
            Angle = 180
        XSpeed = C1Speed*math.cos(math.radians(Angle))
        YSpeed = C1Speed*math.sin(math.radians(Angle))
    C1.speedx = XSpeed
    C1.speedy = YSpeed


def Move():
    for Circle in Circles:
        Circle.x += Circle.speedx
        Circle.y += Circle.speedy

def CollisionDetect():
    for Circle in Circles:
        if Circle.x < Circle.radius or Circle.x > width-Circle.radius:    Circle.speedx *= -1
        if Circle.y < Circle.radius or Circle.y > height-Circle.radius:    Circle.speedy *= -1
    for Circle in Circles:
        for Circle2 in Circles:
            if Circle != Circle2:
                if math.sqrt(  ((Circle.x-Circle2.x)**2)  +  ((Circle.y-Circle2.y)**2)  ) <= (Circle.radius+Circle2.radius):
                    CircleCollide(Circle,Circle2)

def Draw():
    Surface.fill((25,0,0))

    blue = (0,0,150)
    green = (0,150,0)


    for Circle in Circles:
        pygame.draw.circle(Surface,blue,(int(Circle.x),int(height-Circle.y)),Circle.radius)
    for Square in Squares:
        pygame.draw.rect(Surface,green,(int(Square.x),int(Square.y),int(Square.z),int(Square.w)))
    pygame.display.flip()

    global x
    x += 1
    PATH = "/home/cyber/GitHub/pibic/pibic/database/pygame/img/"
    file = PATH + str(x) + '.png'
    pygame.image.save(Surface, file)

def GetInput():
    keystate = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == QUIT or keystate[K_ESCAPE]:
            pygame.quit(); sys.exit()


def main():
    while True:
        GetInput()
        Move()
        CollisionDetect()
        Draw()


if __name__ == '__main__': main()
