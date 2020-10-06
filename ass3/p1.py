import numpy as np
import sys
sys.stdout = open('output/p1_out.txt', 'w')

vals = []
with open("dish_data.txt", "r") as fl:
    while True: ##this is dusgusting dont do it this way i was gonna use a walrus operator but im running python 3.7 and am too lazy to change
        try:
            vals.append([np.float(x) for x in fl.readline().split(" ")])
        except:
            break
vals = np.array(vals)

x = vals[:,0]
y = vals[:,1]
z = vals[:,2]

A = np.zeros((len(z),4))


##expanding out we will find that there is 
## a  for x^2 + y^2
## -2ax_0 for x
## -2ay_0 for y
## ax_0^2 + ay_0^2 + z_0

A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = 1
u,s,v = np.linalg.svd(A,0)

fitp = v.T@(np.diag(1/s)@(u.T @z))
## to go back to the actual things we gotta do some inversion 


actp = np.zeros(4)
actp[0] = fitp[0]
actp[1] = -fitp[1]/(2*fitp[0])
actp[2] = -fitp[2]/(2*fitp[0])
actp[3] = fitp[3] - fitp[0] * fitp[1]**2 - fitp[0] * fitp[2]**2
print("fit params (a, x_0, y_0, z_0) in mm: ", actp)
#with focal length fiven by something like:
print("focal length in mm: ", 0.25/fitp[0] )

#and the errors are:

pred_z = A@fitp
noise = np.std(pred_z - z)
print("The noise in the data in mm: ",noise)
co_var = np.linalg.inv((A.T@A)/ noise) ##uncorolated noise
A_err = np.sqrt(co_var.diagonal())[0]
f_err = A_err/fitp[0] * (0.25/fitp[0]) #error prop as ratio of uncertainty
print("Uncertainty on the focal length in mm: ",f_err)