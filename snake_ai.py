import numpy as np

def get_hamiltonian(n):
    if n%2 != 0:
        return "need even grid size"
    path = np.zeros((n**2,2))
    pos = [0,0]
    direction = 1
    for i in range(n**2):
        path[i,:] = pos
        if direction == 1 and pos[1] == n-1:
            pos[0] += 1
            direction = -1
        elif direction == -1 and pos[1] == 1:
            if pos[0] == n-1: ##go back to the top
                direction = "top"
                pos[1] -= 1
            else:
                pos[0] += 1
                direction = 1
        elif direction == "top":
            if pos[0] == 0:
                return path
            pos[0] -= 1
        else:
            pos[1] += direction
        
    return path

def follow_path(path, pos_snake):
    pos = list(pos_snake[0,:])
    place_in_path = path.tolist().index(pos)
    direction_vec = path[(place_in_path + 1)%len(path)] - path[place_in_path]
    dir_dict = { (0,1) : "right" , (0,-1): "left", (-1,0) :  "up",  (1,0) : "down" }
    return_val = dir_dict[tuple(direction_vec)]

    return return_val