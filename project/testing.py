import numpy as np
from nbody_tools import Nbody
from matplotlib import pyplot as plt


def sim_wrap(nbody_name, steps, frame_rate):
    res = nbody_name.run_sim(steps = steps, frame_return = frame_rate)
    print("Energy changed by:", (np.max(res["energy"]) - np.min(res["energy"]))/np.mean(res["energy"]) * 100, "% throughout the simulation")
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
    print("This movie experience has been brought to you by the Nbody movie company. \nRemeber to take all trash with you on the way out")

def power_law(N, alpha = -3):
    r = np.random.rand(N)
    x = (1-r)**(1/(alpha +1))
    return x


##starting with the 1 particle at rest remains at rest!
if False:
    position = np.array([[0.5,0.5,0.5]])
    mass = np.array([1])
    rest_example = Nbody(pos = position, mass = mass, dimension = 2, grid_ref = 1000)
    sim_wrap(rest_example, 2000, 50)


##now to show that parcticles will orbit
if False:
    positions = np.array([[0.5,0.5],[0.5,0.6]])
    mass = np.ones(2)*5
    vel = np.array([[0.004,0], [-0.004, 0]])
    orbit_example = Nbody(pos = positions, vel = vel, mass = mass, dimension = 2, grid_ref =5550, dt=0.01 )
    sim_wrap(orbit_example, 100000, 100)
    ##this will simulate a stupidly long time into the future to demonstrate that it is fairly good

##big sim circular
if False:
    positions = np.random.rand(100000,2)
    mass = np.ones(100000)
    periodic_example = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =3, dt=0.001)
    sim_wrap(periodic_example, 3000, 25)
    #takes about 2 minute
#big sim non periodic
if False:
    positions = np.random.rand(10000,2)
    mass = np.ones(10000)
    periodic_example = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =3, dt=0.001, BC= "hard")
    sim_wrap(periodic_example, 3000, 25)
    #takes about 30 secs
    #looks exactly as expected lmao....
#big sim with k
if True:
    positions = np.random.rand(100000,2)
    mass = power_law(100000)
    univ_example = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =1, dt=0.001)
    sim_wrap(univ_example, 3000, 25)
    
