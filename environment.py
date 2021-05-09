import pygame
from pygame.locals import USEREVENT
import numpy as np

# CONSTANTS
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SPEED = 10
FPS = 60
GRAVITY = 15
CEILING = -15
FLOOR = 512
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Entity(pygame.sprite.Sprite):
    """Base class for the images we are loading into our game."""
    def __init__(self, x, y, image=None):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.image = pygame.image.load(image)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self):
        """Draw the images after a FPS cycle"""
        SCREEN.blit(self.image, (self.x, self.y), self.rect)


class Background(Entity):
    """Class which handles the background of the game."""
    def __init__(self, x, y, image):
        Entity.__init__(self, x, y, image)
        # rescale the background to be the size of the window
        self.rect.height = SCREEN_HEIGHT
        self.rect.width = SCREEN_WIDTH
        self.image = pygame.transform.scale(
            self.image, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )


class Pipe(Entity):
    """Class which handles the pipes of the game."""
    def update(self):
        self.x -= SPEED


class Bird(Entity):
    """Class that handles the bird which is controlled by the DQN."""
    def __init__(self, x, y, image):
        Entity.__init__(self, x, y, image)
        # rescale
        self.rect.width = 50
        self.rect.height = 40
        self.image = pygame.transform.scale(
            self.image, (self.rect.width, self.rect.height)
        )
        self.jumping = False
        self.jump_speed = 10

    def update(self, action, self_play=False):
        """Updates the bird position and performs action"""
        if action or self_play:
            # if the chosen action is to jump, then we need to either start or
            # reset a jump.
            if self.jumping:
                self.jump_speed = 10
            else:
                self.jumping = True

        if self.jumping:
            if self.jump_speed >= 0:
                jump = (self.jump_speed ** 2) * 0.4

                # if we're not higher than the ceiling, perform a frame
                    # with a partial jump
                if self.y - jump > CEILING:
                    self.y -= jump
                self.jump_speed -= 1
            else:
                # otherwise end the jump
                self.jumping = False
                self.jump_speed = 10

        self.y += GRAVITY

    def collide(self, game_obj):
        """This function handles the collision between bird and the pipes"""
        rect1 = pygame.rect.Rect(self.x, self.y,
                    self.rect.width, self.rect.height)
        rect2 = pygame.rect.Rect(game_obj.x, game_obj.y,
                    game_obj.rect.width, game_obj.rect.height)
        return bool(rect1.colliderect(rect2))


class Environment():
    """This class is responsible for the functionality of the game."""
    def __init__(self):
        # initialize twice, so we're able to create a sliding background
        self.background = Background(0, 0, 'Background.png')
        self.background2 = Background(SCREEN_WIDTH, 0, 'Background.png')
        self.player = Bird(50, 300, 'Bird.png')
        self.pipes = []
        self.done = False
        self.space = False
        self.pipe_spawn = np.random.randint(50, 75)

    def step(self, action, render=True):
        """This function creates a single step in the environment. A step
        is often defined as a few frames following a certain chosen action by
        the DQN."""
        pygame.time.Clock().tick(FPS)
        reward = 0
        self.handle_events()
        self.pipe_spawn -= 1

        # spawn a pipe every 50 - 75 frames
        if self.pipe_spawn == 0:
            self.spawn_pipes()
            self.pipe_spawn = np.random.randint(50, 75)

        ### GAME UPDATES
        self.slide_background()
        self.player.update(action, self.space)

        # pop off-screen pipes
        while self.pipes and self.pipes[0].x <= -self.pipes[0].rect.width:
            self.pipes.pop(0)
            self.pipes.pop(0)
            reward += 5

        for pipe in self.pipes:
            pipe.update()

        # handle collision with the pipes and the floor
        collision = any(self.player.collide(p) for p in self.pipes)
        if collision or self.player.y + self.player.rect.height >= FLOOR:
            reward -= 1
            self.done = True

        ### GAME DRAWING
        if render:
            self.background.draw()
            self.background2.draw()
            self.player.draw()
            for pipe in self.pipes:
                pipe.draw()
            pygame.display.update()

        state = self.get_state()
        self.space = False
        return self.done, state, reward

    def handle_events(self):
        """Handles the pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN:
                # allow for user input
                if event.key == pygame.K_SPACE:
                    self.space = True

    def spawn_pipes(self):
        """Spawns a set of pipes."""
        spawn_height = np.random.randint(220, 490)
        self.pipes.append(
            Pipe(
                SCREEN_WIDTH+100,
                spawn_height,
                'pipeUp.png'
            )
        )
        # dont let the pipe go through the floor
        self.pipes[-1].rect.height = FLOOR - self.pipes[-1].y

        # make sure there is a distance between the pipe spawns
        spawn_height -= np.random.randint(SCREEN_HEIGHT+50,
                                SCREEN_HEIGHT+100)
        self.pipes.append(
            Pipe(
                SCREEN_WIDTH+100,
                spawn_height,
                'pipeDown.png'
            )
        )


    def slide_background(self):
        """Keeps the background looping infinitely, so it looks like we are
        sliding from left to right."""
        self.background.x -= SPEED
        self.background2.x -= SPEED

        if self.background.x <= -SCREEN_WIDTH:
            self.background.x = SCREEN_WIDTH
        if self.background2.x <= -SCREEN_WIDTH:
            self.background2.x = SCREEN_WIDTH

    def get_state(self):
        """This function returns the state of the game in terms of where the
        player approximately is in the game."""
        state = np.zeros(5)
        # center of the player
        center = (self.player.x+self.player.rect.width/2, 
                    self.player.y+self.player.rect.height/2)
        # we dont have to account for the x-value of the bird, this
        # does not change.
        state[0] = center[1]
        if self.pipes:
            # distance in x direction from bird to pipes
            state[1] = self.pipes[0].x - center[0]
            # distance in y direction from bird to upwards pipe
            state[2] = self.pipes[0].y - center[1]
            # distance in y direction from bird to downwards pipe
            state[3] = self.pipes[1].y + self.pipes[1].rect.height - center[1]
            # distance to the end of the pipe in the x direction
            state[4] = self.pipes[0].x + self.pipes[0].rect.width - center[0]

        return state
