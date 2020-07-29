import pygame
from pygame.locals import *
import random, math, sys
import time


# Configurations
STORE_PATH = "../.database/pibic/pygame/img"
FPS = 100
CIRCLE_BUBBLES = 5
SQUARE_BUBBLES = 1
BUBBLES_COLOR = (0,0,150)
SURFACE_COLOR = (25,0,0)
WIDTH, HEIGHT = 50, 50
BUBBLES_RADIUS = 10
# BUBBLES_RADIUS = int(random.random()*50) + 1

# Init Game
pygame.init()

# Set game configurations
surface = pygame.display.set_mode((WIDTH,HEIGHT))
clock = pygame.time.Clock()


class Circle:
    def __init__(self):
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS
        self.x = random.randint(self.radius, WIDTH-self.radius)
        self.y = random.randint(self.radius, HEIGHT-self.radius)
        self.speedx = random.random()
        self.speedy = random.random()
    
    def board_collision(self):
        if self.x < self.radius or self.x > WIDTH-self.radius:
            self.speedx *= -1
        if self.y < self.radius or self.y > HEIGHT-self.radius:
            self.speedy *= -1
    
    def circle_collision(self, other_circle):
        if math.sqrt(((self.x-other_circle.x)**2)+((self.y-other_circle.y)**2)) <= (self.radius+other_circle.radius):
            self.collision()
    
    def collision(self):
        self.speedx *= -1
        self.speedy *= -1
    
    def move(self):
        self.x += self.speedx
        self.y += self.speedy
    
    def show(self):
        pygame.draw.circle(surface, BUBBLES_COLOR, (int(self.x),int(HEIGHT-self.y)), self.radius)


class Square:
    def __init__(self):
        self.color = BUBBLES_COLOR
        self.radius = BUBBLES_RADIUS # Describes the circumference that cover the square
        self.x = random.randint(self.radius, WIDTH-self.radius)
        self.y = random.randint(self.radius, HEIGHT-self.radius)
        self.z = math.sqrt(2*(self.radius**2))
        self.w = math.sqrt(2*(self.radius**2))
        self.speedx = random.random()
        self.speedy = random.random()

        # self.radius = int((math.sqrt(2*(self.side**2))) / 2) # Describes the circumference that cover the square
    
    def board_collision(self):
        if self.x < self.radius or self.x > WIDTH-(2*self.radius):
            self.speedx *= -1
        if self.y < self.radius or self.y > HEIGHT-(2*self.radius):
            self.speedy *= -1
    
    def square_collision(self, other_square):
        if math.sqrt(((self.x-other_square.x)**2)+((self.y-other_square.y)**2)) <= (self.radius+other_square.radius):
            self.collision()
    
    def collision(self):
        self.speedx *= -1
        self.speedy *= -1
    
    def move(self):
        self.x += self.speedx
        self.y += self.speedy

    def show(self):
        pygame.draw.rect(surface, BUBBLES_COLOR,(int(self.x),int(self.y),int(self.z),int(self.w)))


class Draw:
    circles = []
    squares = []

    def __init__(self):
        # for x in range(CIRCLE_BUBBLES):
        #     self.circles.append(Circle())
        for x in range(SQUARE_BUBBLES):
            self.squares.append(Square())
    
    def move_circle(self):
        for circle in self.circles:
            circle.board_collision()

            for other_circle in self.circles:
                if circle != other_circle:
                    circle.circle_collision(other_circle)
            
            circle.move()
            circle.show()

    def move_square(self):
        for square in self.squares:
            square.board_collision()

            for other_square in self.squares:
                if square != other_square:
                    square.square_collision(other_square)
            
            square.move()
            square.show()  

    def show(self):
        pygame.display.flip()
        surface.fill(SURFACE_COLOR)
        clock.tick(FPS)
    
    def save(self):
        pass
        # global x
        # x += 1
        # PATH = "/home/cyber/GitHub/pibic/pibic/database/pygame/img/"
        # file = PATH + str(x) + '.png'
        # pygame.image.save(Surface, file)

    def close(self):
        keystate = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or keystate[K_ESCAPE]:
                pygame.quit(); sys.exit()


def run(draw):
    # for i in range(0,100):
    while True:
        draw.close()
        draw.move_square()        
        draw.show()
    pygame.quit(); sys.exit()


draw = Draw()
if __name__ == '__main__': run(draw)
