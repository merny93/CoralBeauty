import numpy as np
import cmb_utils as ct
import time
from matplotlib import pyplot as plt

import sys
sys.stdout = open('output/p4_out.txt', 'w')

def run_mcmc(pars,data,corr_mat, chifun = ct.my_chi, nstep=500, time_out = None):
    start_time = time.time()
    npar=len(pars)
    chain=[]
    chivec=[]

    chi_cur=chifun(data,pars)

    #generate the weird projection space
    L=np.linalg.cholesky(corr_mat)

    for _ in range(nstep):
        pars_trial=pars+L@np.random.randn(npar)
        if pars_trial[3] < 0:
            pars_trial[3] = pars[3]
        chi_trial=chifun(data,pars_trial)
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            # print("Got new Chi: ", chi_trial)
            pars=pars_trial
            chi_cur=chi_trial
        chain.append(pars)
        chivec.append(chi_cur)
        if time_out is None:
            pass
        elif time.time() - start_time > time_out:
            break
    return np.array(chain),np.array(chivec)



curvature = np.load("output/newton_curvature.npy")
pars = np.load("output/newton_params.npy")
temp_mat = np.zeros((len(pars), len(pars)))
temp_mat[:3,:3] = curvature[:3,:3]
temp_mat[3,3] = pars[3] * 0.01
temp_mat[4:,4:] = curvature[3:,3:]
temp_mat[4:,:3] = curvature[3:,:3]
temp_mat[:3,4:] = curvature[:3,3:]
curvature = temp_mat
wmap=ct.read_wmap()
data=wmap[:,0:3]
chain,chivec=run_mcmc(pars,data,curvature,nstep=1000, time_out=60*60)

plt.clf()
plt.plot(chivec)
plt.savefig("output/chivec1.png")
plt.clf()
plt.plot(chain[:,0])
plt.savefig("output/chain1_0.png")
np.save("output/chain1", chain)
np.save("output/chivec1", chivec)

##this def did not converge just yet so lets do it again with the prior that we have:
delt=chain.copy()
for i in range(delt.shape[1]):
    delt[:,i]=delt[:,i]-delt[:,i].mean()
#by using a covariance matrix to draw trial steps from,
#we get uncorrelated samples much, much faster than
#just taking uncorrelated trial steps
mycov=delt.T@delt/chain.shape[0]
chain2,chivec2=run_mcmc(pars,data,mycov,nstep=5000, time_out=60*60*5)
plt.clf()
plt.plot(chivec2)
plt.savefig("output/chivec2.png")
plt.clf()
plt.plot(chain2[:,0])
plt.savefig("output/chain2_0.png")
np.save("output/chain2", chain2)
np.save("output/chivec2", chivec2)


print("Best params are now: ", np.mean(chain2, axis = 0))
print("with errors given by: ", np.std(chain2, axis = 0))