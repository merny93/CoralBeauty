import numpy as np
from matplotlib import pyplot as plt

import sys
sys.stdout = open('output/p2_out.txt', 'w')

import cmb_utils as ct

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=ct.read_wmap()



print("Chi^2 from jon params is: ", chi_org := ct.my_chi(wmap, pars))
