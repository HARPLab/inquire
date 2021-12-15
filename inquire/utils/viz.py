# Adapted from: https://gist.github.com/KenoLeon/bbbbe02f38e32af53ae134d8fdad0de0

import pdb
import sys
import pygame
import pygame.freetype
import pygame.font
from pygame.locals import KEYDOWN, K_q, K_RIGHT, K_LEFT
from inquire.interactions.feedback import Trajectory
import numpy as np

# CONSTANTS:
SCREENSIZE = WIDTH, HEIGHT = 800, 600
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (160, 160, 160)

_VARS = {'surf': False, 'gridWH': 400,
         'gridOrigin': (200, 100), 'lineWidth': 2}

class Viz:
    def __init__(self, traj):
        state = traj[0][1]
        self.start = state[0]
        self.end = state[1]
        self.puddle_map = state[2]
        self.grid_dim = self.puddle_map.shape[0]
        self.draw_step = 0
        self.traj = traj
        pygame.init()
        self.text_font = pygame.font.SysFont(None, 62)
        _VARS['surf'] = pygame.display.set_mode(SCREENSIZE)
        self.exit = False

    def draw(self):
        self.checkEvents()
        if self.exit:
            return
        _VARS['surf'].fill(GREY)
        self.drawSquareGrid(
         _VARS['gridOrigin'], _VARS['gridWH'], self.grid_dim)
        self.placeCells()
        self.draw_current_position()
        pygame.display.update()

    def draw_current_position(self):
        if self.traj is not None:
            self.draw_step = min(len(self.traj)-1, max(0, self.draw_step))
            step = self.traj[self.draw_step][1][0]
            column = step[0]
            row = step[1]

            cellBorder = 6
            celldimX = celldimY = (_VARS['gridWH']/self.grid_dim) - (cellBorder*2)
            x = _VARS['gridOrigin'][0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + _VARS['lineWidth']/2
            y = _VARS['gridOrigin'][1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + _VARS['lineWidth']/2

            img = self.text_font.render("X", True, BLUE)
            rect = img.get_rect()
            rect.topleft = (x, y)
            _VARS["surf"].blit(img, rect)

    def placeCells(self):
        # GET CELL DIMENSIONS...
        cellBorder = 6
        celldimX = celldimY = (_VARS['gridWH']/self.grid_dim) - (cellBorder*2)
        # DOUBLE LOOP
        for row in range(self.puddle_map.shape[0]):
            for column in range(self.puddle_map.shape[1]):
                # Is the grid cell tiled ?
                if(self.puddle_map[column][row] == 1):
                    color = RED
                elif(self.start == [column,row]):
                    color = GREEN
                elif(self.end == [column,row]):
                    color = BLACK
                else:
                    color = None
                if color is not None:
                    self.drawSquareCell(
                        _VARS['gridOrigin'][0] + (celldimY*row)
                        + cellBorder + (2*row*cellBorder) + _VARS['lineWidth']/2,
                        _VARS['gridOrigin'][1] + (celldimX*column)
                        + cellBorder + (2*column*cellBorder) + _VARS['lineWidth']/2,
                        celldimX, celldimY, color)

    # Draw filled rectangle at coordinates
    def drawSquareCell(self, x, y, dimX, dimY, color, img=_VARS['surf']):
        pygame.draw.rect(
         _VARS['surf'], color,
         (x, y, dimX, dimY)
        )


    def drawSquareGrid(self, origin, gridWH, cells):

        CONTAINER_WIDTH_HEIGHT = gridWH
        cont_x, cont_y = origin

        # DRAW Grid Border:
        # TOP lEFT TO RIGHT
        pygame.draw.line(
          _VARS['surf'], BLACK,
          (cont_x, cont_y),
          (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), _VARS['lineWidth'])
        # # BOTTOM lEFT TO RIGHT
        pygame.draw.line(
          _VARS['surf'], BLACK,
          (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
          (CONTAINER_WIDTH_HEIGHT + cont_x,
           CONTAINER_WIDTH_HEIGHT + cont_y), _VARS['lineWidth'])
        # # LEFT TOP TO BOTTOM
        pygame.draw.line(
          _VARS['surf'], BLACK,
          (cont_x, cont_y),
          (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), _VARS['lineWidth'])
        # # RIGHT TOP TO BOTTOM
        pygame.draw.line(
          _VARS['surf'], BLACK,
          (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
          (CONTAINER_WIDTH_HEIGHT + cont_x,
           CONTAINER_WIDTH_HEIGHT + cont_y), _VARS['lineWidth'])

        # Get cell size, just one since its a square grid.
        cellSize = CONTAINER_WIDTH_HEIGHT/cells

        # VERTICAL DIVISIONS: (0,1,2) for grid(3) for example
        for x in range(cells):
            pygame.draw.line(
               _VARS['surf'], BLACK,
               (cont_x + (cellSize * x), cont_y),
               (cont_x + (cellSize * x), CONTAINER_WIDTH_HEIGHT + cont_y), 2)
        # # HORIZONTAl DIVISIONS
            pygame.draw.line(
              _VARS['surf'], BLACK,
              (cont_x, cont_y + (cellSize*x)),
              (cont_x + CONTAINER_WIDTH_HEIGHT, cont_y + (cellSize*x)), 2)


    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_q:
                pygame.display.quit()
                self.exit = True
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                self.draw_step += 1
            elif event.type == KEYDOWN and event.key == K_LEFT:
                self.draw_step -= 1


if __name__ == '__main__':
    start = [1,5]
    end = [4,6]
    puddles = np.array([
        [0,0,0,0,0,0,0,0],
        [1,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,1,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,1,0,1,1,0,0,1],
        [0,0,0,1,0,0,1,0],
        [0,0,0,0,0,1,1,1]])
    state = [None, [start, end, puddles]]
    viz = Viz([state])
    while not viz.exit:
        viz.draw()

