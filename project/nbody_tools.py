import numpy as np
import scipy as sp
from numba import njit
from matplotlib import pyplot as plt
import warnings

def r_squared(r,k):
    return k/r

@njit
def fast_hash(grid_indexii, grid_shape, mass, grid):
    for particle_num, grid_index in enumerate(grid_indexii):
        ##loop through all the particles:
        grid[grid_index[0], grid_index[1]] += mass[particle_num]
    return grid


class Nbody:
    def __init__(self, pos = None, mass = None, potential = r_squared, dimension = 3, BC = "wrap", grid_size = 1, grid_ref = 1):
        if mass is None or pos is None:
            raise ValueError
        else:
            assert(pos.shape[0] == mass.size)
            assert(pos.shape[1] == dimension)
        self.pos = pos
        self.m = mass
        self.dim = dimension
        self.BC = BC
        self.grid_size = grid_size
        self.ref = np.power(grid_size**self.dim/(self.m.size*grid_ref), 1/self.dim)
        self.grid = self.hash()
        self.pot_func = potential
        self.pot_template = self.make_template()
        self.pot_template = self.make_template_ft()
    def pos_to_grid(self, pos):
        return np.floor(pos/self.ref).astype(int)
    def hash(self):
        ##this will do the heavy lifting of reporting how much mass is in each grid square:
        N = int(np.ceil(self.grid_size/self.ref)) 
        grid_shape = [N for _ in range(self.dim)]
        grid = np.zeros(grid_shape)
        grid_indexii = self.pos_to_grid(self.pos) #np.floor(self.pos/self.ref).astype(int)
        # for particle_num, grid_index in enumerate(grid_indexii):
        #     ##loop through all the particles:
        #     grid[grid_index[0], grid_index[1]] += self.m[particle_num]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hash_grid = fast_hash(grid_indexii,grid_shape,self.m, grid)
        if self.BC == "wrap":
            return hash_grid
        elif self.BC == "hard":
            print("need to pad"):
            raise ValueError
            return
        
    def make_template(self):
        if self.BC == "wrap":
            template = np.zeros_like(self.grid, dtype=float)
            idx = np.roll(np.arange(-template.shape[0]//2, template.shape[0]//2, dtype=float), template.shape[0]//2)
            basis_vec = np.array([idx for _ in range(self.dim)])
            dist_grid = np.array(np.meshgrid(*basis_vec))
            
            dist_grid[:,0,0] = 0.5 #softening
            #print(dist_grid)
            template = self.pot_func(np.linalg.norm(dist_grid, axis=0, keepdims=False), 1)
            #print(template.shape)
            return template
        elif self.BC == "hard":
            print("not implemented yet")
            raise ValueError
            return
    def make_template_ft(self):
        return  sp.fft.rfftn(self.pot_template)
    def get_force(self, particle_pos, pot):
        if self.BC == "wrap":
            #delete potential due to me
            grid_pos = self.pos_to_grid(particle_pos)
            part_pot = pot - np.roll(self.pot_template, tuple(grid_pos))
            #use ballanced derivative
            temp_basis = np.zeros(self.dim, dtype=int)
            temp_basis[0]=1
            basis = [np.roll(temp_basis, x) for x in range(self.dim)]
            derives = np.zeros(self.dim)
            for dir,vec in enumerate(basis):
                derives[dir] = (part_pot[tuple((grid_pos + vec)%part_pot.shape)] - part_pot[tuple((grid_pos + vec)%part_pot.shape)])/self.ref
            return derives
        elif self.BC == "hard":
            print("not there yet")
            raise ValueError
            return
    def make_pot(self)
        #all u gotta do is convolve hopefully bc is handled for me]
        mass_ft = sp.fft.rfftn(self.grid)
        my_pot_ft = mass_ft * self.pot_template_ft
        return sp.fft.irfftn(my_pot_ft)

    def step_pos(self):
        #compute the new positions from the velocity and then rehash the density map
        pass
    
    def step_vel(self):
        #compute a new potential and calculate the force to step velocity
        pass
    
    
    def plot_heatmap_2D(self):
        if self.dim != 2:
            return
        plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        plt.subplot(2, 1, 1)
        plt.imshow(self.grid, origin='lower')
        plt.subplot(2, 1, 2)
        plt.scatter(self.pos[:,1], self.pos[:,0])
        plt.ylim(0, self.grid_size)
        plt.xlim(0, self.grid_size)
        plt.show()
        return

