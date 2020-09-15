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


blue = (0,0,254)
red = (254, 0, 0)
green = (0, 254, 0)

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

##now the first piece of food
board, food_pos = spawn_food(board)

##we might be able to get away without the board explcitly and use player_pos and food_pos as a sort of hash table 

screen = pg.display.set_mode(resolution)
pg.display.set_caption("Snakey")

clock = pg.time.Clock()

crashed = False 

while not crashed:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            crashed = True
        
        print(event)
    
    pg.display.update()
    clock.tick(2)


pg.quit()
quit()