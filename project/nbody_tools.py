import numpy as np
import scipy as sp
from numba import njit
import time

import matplotlib.animation as animation
from matplotlib import pyplot as plt
import warnings



def r_squared(r,k):
    """
    Potential used in problem
    """
    return -k/r

@njit
def fast_hash(grid_indexii, grid_shape, mass, grid):
    """
    Does a numba particle postion to grid desnity
    """
    for particle_num, grid_index in enumerate(grid_indexii):
        ##loop through all the particles:
        grid[grid_index[0], grid_index[1]] += mass[particle_num]
    return grid



class Nbody:
    """
    The big boy Nbody solver class which uses FFTs to solve in nlogn time

    Init a problem by providing the positions, velocities and masses of the particles (as well as some extra thing check the __init__ function)

    Run a simulation by calling run sim.
    """
    def __init__(self, pos = None, vel = None, mass = None, potential = r_squared, dimension = 2, BC = "wrap", grid_size = 1, grid_ref = 1, dt=0.01):
        """
        Take all the inputs and store them and do some precomputation work
        """
        if mass is None or pos is None:
            raise ValueError
        else:
            assert(pos.shape[0] == mass.size)
            assert(pos.shape[1] == dimension)
            if vel is None:

                self.vel = np.zeros_like(pos)
            else:
                assert(pos.shape == vel.shape)
                self.vel = vel
        
        self.pos = pos
        self.m = mass
        self.dim = dimension
        self.BC = BC
        self.grid_size = grid_size

        #this is grid refinement, where it is designed to have about 1 particle per grid_ref
        self.ref = np.power(grid_size**self.dim/(self.m.size*grid_ref), 1/self.dim)
        self.grid = self.hash()
        self.pot_func = potential
        self.pot_template = self.make_template()
        self.pot_template_ft = self.make_template_ft()
        self.dt= dt

    def pos_to_grid(self, pos):
        """
        Simple helper that converts from coordinate to grid square index
        """
        return np.floor(pos/self.ref).astype(int)

    def hash(self):
        """
        main position to grid mass function, uses the numba thing above to go faster
        """

        #generate the grid
        N = int(np.ceil(self.grid_size/self.ref)) 
        grid_shape = [N for _ in range(self.dim)]
        grid = np.zeros(grid_shape)

        #grid indexes from coordinates
        grid_indexii = self.pos_to_grid(self.pos)
        
        ##something about numba changing some dependancies and it dropping a warning which i couldnt be bothered to fix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #call the actual hash function
            hash_grid = fast_hash(grid_indexii,grid_shape,self.m, grid)
        if self.BC == "wrap":
            #great its already circular so return
            return hash_grid
        elif self.BC == "hard":
            ##we need to pad
            print("need to pad")
            raise ValueError
            #return
        
    def make_template(self):
        """
        A very tryhard function which generates the potential centered around a corner
        """
        if self.BC == "wrap":
            #make the zeros template
            template = np.zeros_like(self.grid, dtype=float)

            #generate a basis vector (this will not handle a non-square grid)
            #said basis vector is rolled to have zero at index zero and then positive up to half way point and negative thereafter
            idx = np.roll(np.arange(-template.shape[0]//2, template.shape[0]//2, dtype=float), template.shape[0]//2)

            #we need dimension many basis vectors
            basis_vec = np.array([idx for _ in range(self.dim)])

            #this now created the distance grid from all our basis vectors 
            dist_grid = np.array(np.meshgrid(*basis_vec))
            
            #replace the 0 in the top left corner
            dist_grid[:,0,0] = 0.5 #softening

            #generate the template using the potential function
            #this is greens funciton
            template = self.pot_func(np.linalg.norm(dist_grid, axis=0, keepdims=False), 1)
 
            return template
        elif self.BC == "hard":
            print("not implemented yet")
            raise ValueError
            #return
    def make_template_ft(self):
        return  sp.fft.rfftn(self.pot_template)
    def make_pot(self):
        """
        Generate the potential from the mass distribution by convolving 
        The BC should already be handled
        """
        mass_ft = sp.fft.rfftn(self.grid)
        my_pot_ft = mass_ft * self.pot_template_ft
        return sp.fft.irfftn(my_pot_ft,s = self.pot_template.shape)

    def step_pos(self, half_step = False):
        """
        Step the position with provisions for a euleur half step to get things going at the start
        """
        #compute the new positions from the velocity and then rehash the density map
        if half_step:
            self.pos = self.pos + self.vel*self.dt/2
            if self.BC == "wrap":
                #do the wrap around if particles left the grid
                self.pos = self.pos%self.grid_size
            else:
                print("need to implement walls")
                raise ValueError
            #now that we have new position rehash
            self.grid = self.hash()
            return
        self.pos = self.pos + self.vel*self.dt
        ##for CIcrucla rBC only
        if self.BC == "wrap":
            #do the wrap around if particles left the grid
            self.pos = self.pos%self.grid_size
        else:
            print("need to implement walls")
            raise ValueError
        #now that we have new position rehash
        self.grid = self.hash()
        return
    
    def step_vel(self):
        """
        This is to compute the velocity from the field. A bit harder than the last 
        We compute the vector field everywhere first and then evaluate the force individually
        """
        #compute a new potential and calculate the force to step velocity
        #this function does the convolution
        potential = self.make_pot()
        
        #me messing with things
        self.pot_debug = potential

        ##now get force field:
        #we need to roll a little to the right and a little to the left and take the difference
        #this is a gradient operation which pushes it to another dimension for the different components

        #generate an array which is Ndim x grid_side^Ndim
        derivatives = np.zeros([self.pos.shape[1]] + list(self.grid.shape))

        #ok so this one is a bit of a crazy one. this opp array is gonna be a vector [grid_side^Ndim-1, grid_size^Ndim-2, ..., 1]
        #it willl later be used for a crazy broadcasting operation
        from functools import reduce
        opp= np.array([reduce(lambda x,y: x*y, derivatives.shape[i:-1]) for i in range(1, len(derivatives.shape)-1)] + [1])

        #get grid indexes
        grid_pos = self.pos_to_grid(self.pos)

        #this computes the gradiant stacking the dx,dy,dz... in axis=0 of derivatives
        for dim in range(self.pos.shape[1]):
            derivatives[dim, :] = (np.roll(potential, -1, axis=dim) - np.roll(potential, 1, axis=dim))/(2*self.grid_size) 
        #now for magic

        #now we flatten the first axis such that it is Ndim by however many (so flatten the coordinate)
        der_flat = derivatives.reshape(derivatives.shape[0], -1)

        #get the new flat index by dotting the grid position with the opp vector which converst to this new view
        indexes = np.dot(grid_pos, opp.T)

        #index the derivatives by the points of interest
        ordered_der = der_flat[:,indexes]

        #they happen to end up in a horizontal array which i dont like
        ordered_der = np.transpose(ordered_der)

        #update finally.....
        self.vel = self.vel - ordered_der*self.dt /self.m[:, np.newaxis] 
        return
    
    
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

    def run_sim(self, steps = 10000):
        import time
        #print(self.pos)
        t1 = time.time()
        self.step_pos(half_step=True)
        #print(self.pos)
        for hello in range(steps):
            #print(hello)
            t3=time.time()
            self.step_vel()
            t4=time.time()
            self.step_pos()
            t5=time.time()
            #print(t4-t3,t5-t4)
            if hello%50 == 0:
                #print(self.vel)
                # print(self.vel)
                print(self.pos)
                self.plot_heatmap_2D()
                # plt.imshow(self.pot_debug)
                pass
        t2 = time.time()
        print((t2-t1)/100)
