import numpy as np
from matplotlib import pyplot as plt

import sys
sys.stdout = open('output/p3_out.txt', 'w')

import cmb_utils as ct

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=ct.read_wmap()


##define a numerical derivative method
def num_deriv(fun,x,pars,dpar):
    #calculate numerical derivatives of 
    #a function for use in e.g. Newton's method or LM
    derivs=np.zeros([len(x),len(pars)])
    for i in range(len(pars)):
        pars2=pars.copy()
        pars2[i]=pars2[i]+dpar[i]
        f_right=fun(x,pars2)
        pars2[i]=pars[i]-dpar[i]
        f_left=fun(x,pars2)
        derivs[:,i]=(f_right-f_left)/(2*dpar[i])
    return derivs



Ninv= np.diag(1/(wmap[:,2]**2)) #noise matrix
y = wmap[:,1]
x = wmap[:,0]
dpar = pars*1e-2 


print("Now we do a Newthons method solver")

#initial guess
chi_org = ct.my_chi(wmap, pars)
##lets keep track of the derivatives:
derivatives = []
parameters = []
for i in range(10):
    #get val where we are
    model=ct.fit_wrap(x,pars)

    #linearize around that point and save the derivatives 
    derivs=num_deriv(ct.fit_wrap,x,pars,dpar)
    derivatives.append(derivs)

    #get rid of tau cause we aint changing that
    derivs = np.delete(derivs, 3,1)
    
    #how far from the truth
    resid=y-model
    #solve the linear model
    lhs=derivs.T@Ninv@derivs #this is curvature
    rhs=derivs.T@Ninv@resid 
    lhs_inv=np.linalg.inv(lhs)
    step=lhs_inv@rhs #project solution and move over
    
    #add the tau back in as a zero
    step = np.insert(step, 3, 0)
    pars=pars+step
    parameters.append(pars)

    ##calculate the next derivative step as 10% of the error
    dpar = np.insert(np.sqrt(np.diag(lhs_inv))*0.1, 3, dpar[3])

    print("On the: " , i+1, " itteration and params are: ", pars)
    #this next step is not really needed but its nice to see how chi^2 changes
    print("Chi^2 is now: ", chi_now := ct.my_chi(wmap, pars) )
    if np.abs(chi_now - chi_org) < 0.1:
        print("we have converged")
        print("with errors given by :", np.sqrt(np.diag(lhs_inv)))
        break
    chi_org = chi_now

#derivatives = np.array(derivatives)
#parameters = np.array(parameters)

##save the derivatives: they look sensible especially at the end
plt.clf()
plt.plot(derivatives[-1][:,0])
plt.savefig("output/derH0")
plt.clf()
plt.plot(derivatives[-1][:,1])
plt.savefig("output/der_barion_density")
plt.clf()
plt.plot(derivatives[-1][:,2])
plt.savefig("output/der_cold_dark_density")
plt.clf()
plt.plot(derivatives[-1][:,4])
plt.savefig("output/der_premordial")
plt.clf()
plt.plot(derivatives[-1][:,5])
plt.savefig("output/der_power_law")


##lets also save the last parameters so we can use them as a starting point
##we also want to save the curvature matrix so we can have a good step size estimate
np.save("output/newton_params", parameters[-1])
np.save("output/newton_curvature", lhs_inv)
