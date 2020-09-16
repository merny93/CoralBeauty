import pygame as pg
import numpy as np
import warnings

arrows = {"up": "up", "down": "down", "left": "left", "right": "right"}
wasd = {"w": "up", "a": "left", "s": "down", "d":"right"}
blue = (0,0,254)
red = (254, 0, 0)
green = (0, 254, 0)
black = (0,0,0)

def draw_square(cor, surf, col, rect_size):
    rect = pg.Rect(cor[1]*rect_size, cor[0]*rect_size, rect_size, rect_size)
    pg.draw.rect(surf, col, rect)
    return

def draw_player_init(player_pos_list, surf, col, rect_size):
    for pos in player_pos_list:
        draw_square(pos, surf,col,rect_size)
    return
class Game_Point:
    def __init__(self, player, food):
        self.p = player
        self.f = food

class Player_Direction:
    def __init__(self, dir):
        self.set = False
        if dir is "right":
            self.d = [0,1]
        elif dir is "left":
            self.d = [0,-1]
        elif dir is "up":
            self.d = [-1,0]
        elif dir is "down":
            self.d = [1,0]
    def turn(self, dir):
        if self.set == True:
            #already set
            return
        if dir is "right" and not (self.d == [0,-1]): 
            self.d = [0,1]
            self.set = True
        elif dir is "left" and not ( self.d == [0,1]):
            self.d = [0,-1]
            self.set = True
        elif dir is "up" and not (self.d == [1,0]):
            self.d = [-1,0]
            self.set = True
        elif dir is "down" and not (self.d == [-1, 0]):
            self.d = [1,0]
            self.set = True
    def re_set(self):
        self.set = False

class Player:
    def __init__(self, init_length, head_pos, direct, player_color, scrn, rect_size, back_color, controls = None):
        
        if controls is None:
            self.controlled = False
        else:
            self.controlled = True
            self.keys = controls
        self.color = player_color
        self.screen = scrn
        self.rect_size = rect_size
        self.back_color = back_color

        self.pos = np.zeros((init_length, 2), dtype=int)
        if direct not in ["left", "right", "up", "down"]:
            direct= "right"
            warnings.warn("defaukted to right", RuntimeWarning)
        
        if direct is "right":
            self.pos[:,1] = np.arange(head_pos[1], head_pos[1]-init_length, -1)
            self.pos[:,0] = head_pos[0]
            self.dir = Player_Direction(direct)
        elif direct is "left":
            self.pos[:,1] = np.arange(head_pos[1], head_pos[1]+init_length)
            self.pos[:,0] = head_pos[0]
            self.dir = Player_Direction(direct)
        elif direct is "up":
            self.pos[:,0] = np.arange(head_pos[0], head_pos[0]+init_length)
            self.pos[:,1] = head_pos[1]
            self.dir = Player_Direction(direct)
        elif direct is "down":
            self.pos[:,0] = np.arange(head_pos[0], head_pos[0]-init_length, -1)
            self.pos[:,1] = head_pos[1]
            self.dir = Player_Direction(direct)

        ##add a variable unqiue to be deleted 
        self.to_del = [0,0]

    def step(self):
        
        p_pos_temp = np.zeros_like(self.pos)
        p_pos_temp[1:,:] = self.pos[:-1,:]
        p_pos_temp[0,:] = p_pos_temp[1,:] + self.dir.d
        self.dir.re_set()
        self.to_del = self.pos[-1,:]
        self.pos = p_pos_temp

    def add_block(self):
        self.pos = np.append(self.pos, [self.to_del], axis = 0)
        self.to_del = None

    def init_render(self):
        draw_player_init(self.pos, self.screen, self.color, self.rect_size)

    def render_step(self):
        if self.to_del is not None:
            draw_square(self.to_del, self.screen, self.back_color, self.rect_size)
        draw_square(self.pos[0,:], self.screen, self.color, self.rect_size)
    
    def control(self, event):
        if event.type == pg.KEYDOWN:
            if pg.key.name(event.key) in self.keys:
                self.dir.turn(self.keys[pg.key.name(event.key)])


class Food:
    def __init__(self, col, scrn, rect_size):
        self.pos = [0,0]
        self.color = col 
        self.screen = scrn 
        self.rect_size = rect_size

    def spawn(self, players, board_size):
        available_points = [[x,y] for x in range(board_size[0]) for y in range(board_size[1])]
        for player in players:
            for pos in player.pos:
                available_points.remove(list(pos))
        spawn_choice = np.random.choice(len(available_points))
        spawn_point = available_points[spawn_choice]
        self.pos =  spawn_point
    def render(self):
        draw_square(self.pos, self.screen, self.color, self.rect_size)

class Board:
    def __init__(self, board_size, background_color, rect_size, food_color = green):
        self.size = board_size
        self.back_color = background_color
        self.f_color = food_color
        self.rect_size = rect_size
        self.players = []
        self.p_control = 0
        self.grid = [[] for i in range(board_size[0])]
        for i in range(board_size[0]):
            self.grid[i] = [Game_Point(False, False) for i in range(board_size[1])]
    
    def get_res(self):
        resolution = tuple([x * self.rect_size for x in self.size])
        return resolution

    def init_screen(self, scrn):
        self.screen = scrn

    def init_player(self, pos = None, color= red, direct= "right", length= 4, controls = arrows):
        if pos is None:
            pos = [int(self.size[x]/2) for x in range(2)]
        self.players.append(Player(length, pos, direct, color,self.screen, self.rect_size, self.back_color, controls=controls))
        self.players[-1].init_render()

    def spawn_food(self):
        self.food = Food(self.f_color, self.screen, self.rect_size)
        self.food.spawn(self.players,self.size)
        self.food.render()

    def check_food(self):
        for player in self.players:
            head = player.pos[0,:]
            if list(head) == list(self.food.pos):
                player.add_block()
                self.spawn_food()
     
    def check_crash(self):
        flag = False
        for player in self.players:
            for ref in self.players:
                if id(player) == id(ref):
                    list_blocks = ref.pos[1:,:]
                else:
                    list_blocks = ref.pos
                head = player.pos[0,:]
                
                if list(head) in list_blocks.tolist():
                    flag = True
                if np.any(head<0):
                    flag = True
                if head[0] >= self.size[0]:
                    flag = True
                if head[1] >= self.size[1]:
                    flag = True
        return flag


board_size = (40,40)
rect_size = 20
init_size = 4//2

game_board = Board(board_size, black, rect_size)




##we might be able to get away without the board explcitly and use player_pos and food_pos as a sort of hash table 

screen = pg.display.set_mode(game_board.get_res())
pg.display.set_caption("Snakey")

game_board.init_screen(screen)


clock = pg.time.Clock()

crashed = False 

game_board.init_player()
game_board.init_player(pos = [10,10], color = blue, direct= "right",length=4, controls = wasd)
game_board.spawn_food()

while not crashed:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            crashed = True
        for player in game_board.players:
            if player.controlled:
                player.control(event)
    ##update player position graphically

    for player in game_board.players:
        player.step()
        player.render_step()
    game_board.check_food()
    if game_board.check_crash():
        print("***************** Scores: *************")
        for player in game_board.players:
            print(player.color, "got:", player.pos.shape[0])
        crashed = True
    
    pg.display.update()
    clock.tick(5)


pg.quit()
quit()