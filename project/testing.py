import numpy as np
from nbody_tools import Nbody
from matplotlib import pyplot as plt
positions = np.random.rand(10000,2)
#positions = np.array([[0.5,0.5],[0.5,0.6]])
mass = np.ones(10000)

#mass = 1/np.random.power(4, 100000)
#mass = np.random.randn(10000)
#mass = np.ones(2)
#vel = np.array([[0.15,0], [-0.15, 0]])
my_object = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =100 )
#my_object.plot_heatmap_2D()

# import matplotlib.pyplot as plt
# plt.clf()
# plt.imshow(my_object.pot_template)
# plt.show()

results = my_object.run_sim()

for pic in results["density"]:
    plt.ion()
    plt.imshow(pic)
    plt.show()
    plt.pause(0.1)
    plt.clf()




"""
#2 particles in orbit
positions = np.array([[0.5,0.5],[0.5,0.6]])
mass = np.ones(100)
mass = np.ones(2)
vel = np.array([[0.15,0], [-0.15, 0]])
my_object = Nbody(pos = positions, vel = vel, mass = mass, dimension = 2, grid_ref =5550, dt=0.001 )
#my_object.plot_heatmap_2D()

# import matplotlib.pyplot as plt
# plt.clf()
# plt.imshow(my_object.pot_template)
# plt.show()

results = my_object.run_sim(steps = 10000, frame_return = 100)

for pic in results["density"]:
    plt.ion()
    plt.imshow(pic)
    plt.show()
    plt.pause(0.1)
    plt.clf()

"""