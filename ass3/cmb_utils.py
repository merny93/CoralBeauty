import numpy as np
from sky_exmp import fit_wrap

def my_chi(wmap, pars, func = fit_wrap): ##get chi^2 for fun and profit
    return np.sum((fit_wrap(wmap[:,0],pars) - wmap[:,1])**2 / wmap[:,2]**2)

def read_wmap(file_name= "sky_data.txt", end_point = "/", comment = "#"):
    vals = []
    with open(file_name, "r") as fl:
        while ((line := fl.readline())[0] != end_point): ##guess who installed python3.8
            if line[0] == comment:
                continue
            vals.append([float(x) for x in line.split()])
    vals = np.array(vals)
    return vals   
