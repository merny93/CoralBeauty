import numpy as np
from nbody_tools import Nbody

positions = np.random.rand(10000,2)
positions = np.array([[0.1,0.1],[0.5,0.5]])
mass = np.ones(10000)
mass = np.ones(2)
my_object = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =150 )
#my_object.plot_heatmap_2D()

# import matplotlib.pyplot as plt
# plt.clf()
# plt.imshow(my_object.pot_template)
# plt.show()

my_object.run_sim()