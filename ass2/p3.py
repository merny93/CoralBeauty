from scipy.integrate import solve_ivp
##magic 

import numpy as np

half_lifes = [4.468e9, 24.1/365, 6.7/24/365,245500, 75380, 1600, 3.8235/365, 3.10/60/24/365, 26.8/60/24/365, 19.9/60/24/365,164e-6 /60/24/365, 22.3, 5.015, 138.376/365]
half_lifes = np.array(half_lifes)

def ur_anium_decay(x, y, half_lifes = half_lifes):
    dydx =  np.zeros(len(half_lifes) + 1)
    for i in range(half_lifes.size):
        dydx[i] = -y[i]/half_lifes[i]
    for i in range(half_lifes.size):
        dydx[i+1] += y[i]/half_lifes[i]
    return dydx




##we need an ic
y0 = np.zeros(len(half_lifes) + 1)
y0[0] = 1
x0 = 0
x1 = np.max(half_lifes)*3

ans_stiff = solve_ivp(ur_anium_decay, [x0,x1], y0, method='Radau') ##obviouslty this is stiff so we should use the implicit technique


import matplotlib.pyplot as plt

plt.plot(ans_stiff.t, ans_stiff.y[0,:], label="Uranium 238")
plt.plot(ans_stiff.t, ans_stiff.y[-1,:], label = "Lead")
plt.xlabel("Time in years")
plt.ylabel("Relative quantity")
plt.title("Uranium to lead")
plt.legend()
plt.show()
##clearly the uranium 238 effectivly instantly turns into lead so the majority of the decay is either in uranium 238 or in lead


plt.plot(ans_stiff.t[:], ans_stiff.y[3,:], label="Uranium 234")
plt.plot(ans_stiff.t[:], ans_stiff.y[4,:], label = "Thorium 230")
plt.xlabel("Time in years")
plt.ylabel("Relative quantity")
plt.title("Thorium and Uranium")
plt.legend()
plt.show()
