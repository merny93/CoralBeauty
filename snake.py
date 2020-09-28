import pygame as pg
import numpy as np
import warnings
import snake_ai as ai
from snake_helper import Board

arrows = {"up": "up", "down": "down", "left": "left", "right": "right"}
wasd = {"w": "up", "a": "left", "s": "down", "d":"right"}
blue = (0,0,254)
red = (254, 0, 0)
green = (0, 254, 0)
black = (0,0,0)


board_size = (20,20)
rect_size = 20
init_size = 4//2

game_board = Board(board_size, black, rect_size)


ham_cycle = ai.get_hamiltonian(board_size[0])


##we might be able to get away without the board explcitly and use player_pos and food_pos as a sort of hash table 

screen = pg.display.set_mode(game_board.get_res())
pg.display.set_caption("Snakey")

game_board.init_screen(screen)


clock = pg.time.Clock()

crashed = False 

game_board.init_player(length = 4, p_name="Simon", controls = None, pos = [0,5])

#game_board.init_player(pos = [9,9], color = blue, direct= "right",length=4, controls = wasd, p_name = "Bob")
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
            if not player.controlled:
                player.control_queue.append(ai.follow_path(ham_cycle, player.pos))

    for player in game_board.players:
        player.step()
        player.render_step()
    game_board.check_food()
    if game_board.check_crash():
        print("***************** Scores: *************")
        for player in game_board.players:
            print(player.pname, "got:", player.pos.shape[0])
        crashed = True
        break
    
    pg.display.update()
    clock.tick(60)


pg.quit()
quit()