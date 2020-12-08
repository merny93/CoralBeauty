import numpy as np
from nbody_tools import Nbody
from matplotlib import pyplot as plt
positions = np.random.rand(50000,2)
#vel = np.random.randn(1000000,2)/10
#positions = np.array([[0.5,0.5],[0.5,0.6]])
mass = np.ones(50000)
# mass[0] = 1000
# mass[1] = mass[0]
# positions[0,:] = [0.5,0.5]
# positions[1,:] = [0.55,0.5]
#masas = 1/np.random.power(4, 100000)
#mass = np.random.randn(10000)
#mass = np.ones(2)
#vel = np.array([[0.15,0], [-0.15, 0]])
# positions = np.array([[0.5,0.5]])
# mass = np.array([1,1])
# positions = np.array([[0.5,0.5], [0.7,0.5]])
my_object = Nbody(pos = positions, mass = mass, dimension = 2, grid_ref =5)
#my_object.plot_heatmap_2D()

# import matplotlib.pyplot as plt
# plt.clf()
# plt.imshow(my_object.pot_template)
# plt.show()

results = my_object.run_sim()

for pic in results["density"]:
    plt.ion()
    plt.imshow(pic, vmax=2)
    plt.show()
    plt.pause(0.1)
    plt.clf()





#2 particles in orbit
# positions = np.array([[0.5,0.5],[0.5,0.6]])
# mass = np.ones(100)
# mass = np.ones(2)
# vel = np.array([[0.15,0], [-0.15, 0]])
# my_object = Nbody(pos = positions, vel = vel, mass = mass, dimension = 2, grid_ref =5550, dt=0.001 )
# #my_object.plot_heatmap_2D()

# # import matplotlib.pyplot as plt
# # plt.clf()
# # plt.imshow(my_object.pot_template)
# # plt.show()

# results = my_object.run_sim(steps = 10000, frame_return = 100)

# for pic in results["density"]:
#     plt.ion()
#     plt.imshow(pic)
#     plt.show()
#     plt.pause(0.1)
#     plt.clf()

