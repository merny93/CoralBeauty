import numpy as np
from matplotlib import pyplot as plt



import cmb_utils as ct

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=ct.read_wmap()



print("Chi^2 from jon params is: ", chi_org := ct.my_chi(wmap, pars))

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
        break
    chi_org = chi_now

derivatives = np.array(derivatives)
parameters = np.array(parameters)

plt.clf()
plt.scatter(parameters[:,0], derivatives[:,0])
plt.savefig("output/derH0")
plt.clf()
plt.scatter(parameters[:,1], derivatives[:,1])
plt.savefig("output/der_barion_density")
plt.clf()
plt.scatter(parameters[:,2], derivatives[:,2])
plt.savefig("output/der_cold_dark_density")
plt.clf()
plt.scatter(parameters[:,4], derivatives[:,4])
plt.savefig("output/der_premordial")
plt.clf()
plt.scatter(parameters[:,5], derivatives[:,5])
plt.savefig("output/der_power_law")

# ##i Guess we can plot a graph 

# def run_mcmc(pars,data,par_step,chifun,nstep=500):
#     npar=len(pars)
#     chain=np.zeros([nstep,npar])
#     chivec=np.zeros(nstep)
    
#     chi_cur=chifun(data,pars)
#     for i in range(nstep):
#         pars_trial=pars+np.random.randn(npar)*par_step
#         if pars_trial[3] < 0:
#             pars_trial = pars[3]
#         chi_trial=chifun(data,pars_trial)
#         #we now have chi^2 at our current location
#         #and chi^2 in our trial location. decide if we take the step
#         accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
#         if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
#             print("accepted")
#             pars=pars_trial
#             chi_cur=chi_trial
#         chain[i,:]=pars
#         chivec[i]=chi_cur
#     return chain,chivec

# ##next problem
# pars = np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
# par_sigs=pars*0.2
# data=wmap[:,0:3]
# chain,chivec=run_mcmc(pars,data,par_sigs,my_chi,nstep=200)
# par_sigs=np.std(chain,axis=0)
# par_means=np.mean(chain,axis=0)
# plt.plot(chivec)
# plt.title("Chi^2 over the steps")
# plt.savefig("Chidecent.png")

# plt.plot(chain[:,3])
# plt.title("tau steps")
# plt.savefig("tausteps.png")
# print(par_means, par_sigs, np.mean(chivec), np.min(chivec))

# nchain=4
# all_chains=[None]*nchain
# for i in range(nchain):
#     pars_start=par_means+3*par_sigs*np.random.randn(len(pars))
#     chain,chivec=run_mcmc(pars_start,data,par_sigs,my_chi,nstep=250)
#     all_chains[i]=chain


# np.save("allchains.npz", np.array(all_chains))


# for i in range(nchain):
#     for j in range(i+1,nchain):
#         mean1=np.mean(all_chains[i],axis=0)
#         mean2=np.mean(all_chains[j],axis=0)
#         std1=np.std(all_chains[i],axis=0)
#         std2=np.std(all_chains[j],axis=0)
#         print('param difference in sigma is ',(mean1-mean2)/(0.5*(std1+std2)))