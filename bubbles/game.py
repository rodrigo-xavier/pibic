import pygame
from pygame.locals import *
import sys
from circle import Circle
from square import Square
import math, random


class BubblesGame:
    bubbles = []

    def __init__(self, SURFACE_COLOR=(0,0,0), FPS=60, CIRCLE_BUBBLES=1, SQUARE_BUBBLES=1, BUBBLES_COLOR=(255,255,255), BUBBLES_RADIUS=1, WIDTH=50, HEIGHT=50, TRAJECTORY='random', TRAJECTORY_RADIUS=25):
        
        self.surface = pygame.display.set_mode((WIDTH,HEIGHT))
        self.surface_color = SURFACE_COLOR
        self.fps = FPS
        self.trajectory = TRAJECTORY

        for i in range(CIRCLE_BUBBLES):
            x, y = self.build_far(BUBBLES_RADIUS, WIDTH, HEIGHT)
            self.bubbles.append(
                Circle(
                    surface=self.surface,
                    surface_color=self.surface_color,
                    bubbles_color=BUBBLES_COLOR,
                    bubbles_radius=BUBBLES_RADIUS,
                    trajectory=TRAJECTORY,
                    trajectory_radius=TRAJECTORY_RADIUS,
                    width=WIDTH,
                    height=HEIGHT,
                    x=x,
                    y=y,
                )
            )
        for i in range(SQUARE_BUBBLES):
            x, y = self.build_far(BUBBLES_RADIUS, WIDTH, HEIGHT)
            self.bubbles.append(
                Square(
                    surface=self.surface,
                    surface_color=self.surface_color,
                    bubbles_color=BUBBLES_COLOR,
                    bubbles_radius=BUBBLES_RADIUS,
                    trajectory=TRAJECTORY,
                    trajectory_radius=TRAJECTORY_RADIUS,
                    width=WIDTH,
                    height=HEIGHT,
                    x=x,
                    y=y,
                )
            )

    def build_far(self, radius, width, height):
        OFFSET = 5
        far = False
        distance = []

        if not self.bubbles:
            x = random.randint(radius+OFFSET, width-radius-OFFSET)
            y = random.randint(radius+OFFSET, height-radius-OFFSET)
            return x, y

        while not far:
            distance = []
            x = random.randint(radius+OFFSET, width-radius-OFFSET)
            y = random.randint(radius+OFFSET, height-radius-OFFSET)

            for bubble in self.bubbles:
                distance.append(math.sqrt(((x-bubble.x)**2) + ((y-bubble.y)**2)))

            if min(distance) > (2*radius + OFFSET):
                far = True
        
        return x, y

    def random_trajectory(self):
        for bubble in self.bubbles:
            if bubble.check_board_collision():
                bubble.board_collision()
            elif self.have_collision(bubble):
                bubble.elastic_collision(self.take_the_nearest(bubble))

            bubble.move()
            bubble.show()

    def circular_trajectory(self):
        self.bubbles[0].move_circular()
        self.bubbles[0].show()
            
    def square_trajectory(self):
        self.bubbles[0].move_square()
        self.bubbles[0].show()

    def have_collision(self, bubble):
        OFFSET = 3

        newx = bubble.x + bubble.v[0]
        newy = bubble.y + bubble.v[1]

        for nearest in self.bubbles:
            if math.sqrt(((newx-nearest.x)**2)+((newy-nearest.y)**2)) <= (bubble.radius + nearest.radius + OFFSET):
                return bubble.check_collision()
        return False

    def take_the_nearest(self, bubble):
        distance = []

        for nearest in self.bubbles:
            if nearest != bubble:
                distance.append(math.sqrt(((bubble.x-nearest.x)**2)+((bubble.y-nearest.y)**2)))

        return self.bubbles[distance.index(min(distance))]

    def show(self):
        self.surface.fill(self.surface_color)

        if self.trajectory == 'random':
            self.random_trajectory()
        elif self.trajectory == 'circular':
            self.circular_trajectory()
        elif self.trajectory == 'square':
            self.square_trajectory()

        pygame.display.flip()
        pygame.time.Clock().tick(self.fps)
    
    def save(self, n, img_path):
        file = img_path + str(n) + '.png'
        pygame.image.save(self.surface, file)

    def close(self):
        keystate = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or keystate[K_ESCAPE]:
                pygame.quit(); sys.exit()
    
    # 120 == Frames per lap
    def run(self, loop_counter, img_path, frames, save):
        print(frames)
        if save:
            for i in range(1, frames):
                self.close()
                self.show()

                if self.trajectory != 'random' and LAPS % 5 == 0:
                    file_name = str(loop_counter) + "_" + str(i)

                    if i >= 1 and i <= 120:
                        self.save(file_name, img_path)
                    if i >= 151 and i <= 270:
                        self.save(file_name, img_path)
                    if i >= 301 and i <= 420:
                        self.save(file_name, img_path)
                    if i >= 451 and i <= 570:
                        self.save(file_name, img_path)
                else:
                    self.save(i, img_path)
        else:
            print("here")
            for i in range(1, frames):
                self.close()
                self.show()