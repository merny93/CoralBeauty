import pygame as pg
import numpy as np

class Game_Point:
    def __init__(self, player, food):
        self.p = player
        self.f = food
    def set_p(self, player):
        self.p = player
    def set_f(self, food):
        self.f = food

class Player_Direction:
    def __init__(self, dir):
        if dir is "right":
            self.d = [0,1]
        elif dir is "left":
            self.d = [0,-1]
        elif dir is "up":
            self.d = [-1,0]
        elif dir is "down":
            self.d = [1,0]
    def turn(self, dir):
        if dir is "right" and self.d is not [0,-1]: 
            self.d = [0,1]
        elif dir is "left" and self.d is not [0,1]:
            self.d = [0,-1]
        elif dir is "up" and self.d is not [1,0]:
            self.d = [-1,0]
        elif dir is "down" and self.d is not [-1, 0]:
            self.d = [1,0]


def spawn_food(board_):
    available_points = []
    for row, row_list in enumerate(board_):
        for col, pl_obj in enumerate(row_list):
            if not pl_obj.p:
                available_points.append([row,col])
    spawn_choice = np.random.choice(len(available_points))
    spawn_point = available_points[spawn_choice]
    board_[spawn_point[0]][spawn_point[1]].set_f(True)
    return board_, spawn_point

def draw_square(cor, surf, col, rect_size):
    rect = pg.Rect(cor[1]*rect_size, cor[0]*rect_size, rect_size, rect_size)
    pg.draw.rect(surf, col, rect)
    return

def draw_player_init(player_pos_list, surf, col, rect_size):
    for pos in player_pos_list:
        draw_square(pos, surf,col,rect_size)
    return

blue = (0,0,254)
red = (254, 0, 0)
green = (0, 254, 0)
black = (0,0,0)

board_size = (40,40)
rect_size = 20
init_size = 4//2
resolution = tuple([x * rect_size for x in board_size])
print("resolution is", resolution)

board = [[] for i in range(board_size[0])]
for i in range(board_size[0]):
    board[i] = [Game_Point(False, False) for i in range(board_size[1])]

##Now lets generate an array which contains the player positions in a row. Here numpy arrays are nice and not nice cause we kinda will want to append
##every time we eat a new thing

player_pos = np.zeros((init_size*2, 2), dtype=int)

##init the player 
for i in range(init_size * 2):
    pos = int(board_size[1]/2) - init_size + i
    board[int(board_size[0]/2)][pos].set_p(True)
    player_pos[i,:] = [int(board_size[0]/2),pos]

## init direction of motion
player_direction = Player_Direction("left")
##now the first piece of food
board, food_pos = spawn_food(board)

##we might be able to get away without the board explcitly and use player_pos and food_pos as a sort of hash table 

screen = pg.display.set_mode(resolution)
pg.display.set_caption("Snakey")

clock = pg.time.Clock()

crashed = False 

draw_player_init(player_pos, screen, red, rect_size)

while not crashed:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            crashed = True
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT:
                player_direction.turn("left")
            elif event.key == pg.K_RIGHT:
                player_direction.turn("right")
            elif event.key == pg.K_DOWN:
                player_direction.turn("down")
            elif event.key == pg.K_UP:
                player_direction.turn("up")
        print(event)
    ##update player position graphically
    p_pos_temp = np.zeros_like(player_pos)
    p_pos_temp[1:,:] = player_pos[:-1,:]
    p_pos_temp[0,:] = p_pos_temp[1,:] + player_direction.d

    draw_square(player_pos[-1,:], screen, black, rect_size)
    player_pos = p_pos_temp
    draw_square(player_pos[0,:], screen, red, rect_size)
    pg.display.update()
    clock.tick(2)


pg.quit()
quit()