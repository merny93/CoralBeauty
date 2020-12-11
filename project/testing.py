import numpy as np
from nbody_tools import Nbody
from matplotlib import pyplot as plt


def sim_wrap(nbody_name, steps, frame_rate):
    res = nbody_name.run_sim(steps = steps, frame_return = frame_rate)
    print("Energy changed by:", (np.max(res["energy"]) - np.min(res["energy"]))/np.abs(np.mean(res["energy"])) * 100, "% throughout the simulation")
    input("Sim has been computed, press enter to view the movie")
    plt.clf()
    for pic in res["density"]:
        plt.ion()
        while len(pic.shape) > 2:
            pic = np.sum(pic, axis=-1)
        plt.imshow(pic)
        plt.show()
        plt.pause(0.1)
        plt.clf()
    plt.imshow(pic)
    plt.show()      
    print("This movie experience has been brought to you by the Nbody movie company. \nRemeber to take all trash with you on the way out")

def power_law(N, alpha = -3):
    r = np.random.rand(N)
    x = (1-r)**(1/(alpha +1))
    return x


##starting with the 1 particle at rest remains at rest!
if True:
    position = np.array([[0.5,0.5,0.5]])
    mass = np.array([1])
    rest_example = Nbody(pos = position, mass = mass, dimension = 3, grid_ref = 1000)
    sim_wrap(rest_example, 2000, 50)


##now to show that parcticles will orbit
if True:
    positions = np.array([[0.5,0.5,0.5],[0.5,0.6,0.6]])
    mass = np.ones(2)*5
    vel = np.array([[0.04,0.04,0], [-0.04,-0.04, 0]])
    orbit_example = Nbody(pos = positions, vel = vel, mass = mass, dimension = 3, grid_ref =10550, dt=0.01 )
    sim_wrap(orbit_example, 10000, 50)
    ##this will simulate a stupidly long time into the future to demonstrate that it is fairly good

##big sim circular
if True:
    positions = np.random.rand(100000,3)
    mass = np.ones(100000)
    periodic_example = Nbody(pos = positions, mass = mass, dimension = 3, grid_ref =10, dt=0.0002)
    sim_wrap(periodic_example, 700, 10)
    #takes about 2 minute
#big sim non periodic
if True:
    positions = np.random.rand(5000,3)
    mass = np.ones(5000)
    periodic_example = Nbody(pos = positions, mass = mass, dimension = 3, grid_ref =10, dt=0.0002, BC="hard")
    sim_wrap(periodic_example, 700, 10)
    #takes about 2 mins
    #looks exactly as expected lmao....

#big sim with power law distribution 
if True:
    positions = np.random.rand(100000,3)
    mass = power_law(100000)
    univ_example = Nbody(pos = positions, mass = mass, dimension = 3, grid_ref =10, dt=0.00016)
    sim_wrap(univ_example, 700, 10)
    #takes a long long time
    
