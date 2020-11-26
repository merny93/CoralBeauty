import numpy as np
from nbody_tools import Nbody

positions = np.random.rand(25,2)
#positions = np.array([[0,1],[0.0,0.0]])
mass = np.ones(25)

my_object = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =10 )
my_object.plot_heatmap_2D()

import matplotlib.pyplot as plt
plt.clf()
plt.imshow(my_object.pot_template)
plt.show()
