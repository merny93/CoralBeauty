import numpy as np
import cmb_utils as ct
import time
from matplotlib import pyplot as plt

import sys
sys.stdout = open('output/p5_out.txt', 'w')



##alterntivly we can just re-weigh the previous chain:
old_chain = np.load("output/chain2.npy")
weights  = np.exp(-(old_chain[:,3] - 0.0544)**2/0.0073**2) #might need a 0.5 in exponent but u get the idea
print("from old chain:")
print("Best params are now: ", np.sum(old_chain*weights[:,np.newaxis], axis = 0)/np.sum(weights))

sort_ind = np.argsort(old_chain, axis=0)
total = np.cumsum(weights[sort_ind], axis=0)
print(total.shape)
weights_total = np.sum(weights)
lower_bound = np.argmax(total>( weights_total*(1- 0.65)), axis=0)
lower_bound = [(lower_bound[x], x) for x in range(len(lower_bound))]
upper_bound = np.argmax(total>( weights_total*(0.65)), axis = 0)
upper_bound = [(upper_bound[x], x) for x in range(len(upper_bound))]
old_chain.sort(axis = 0)


print("with lower bound for 65/100 confidence: ", [old_chain[x] for x in lower_bound] )
print("with upper bound for 65/100 confidence: ", [old_chain[x] for x in upper_bound])


def run_mcmc(pars,data,corr_mat, chifun = ct.my_chi, nstep=500, time_out = None):
    start_time = time.time()
    npar=len(pars) ##ignore
    chain=[]
    chivec=[]
    tau = 0.0544
    chi_cur=chifun(data,np.insert(pars,3,tau))

    #generate the weird projection space
    L=np.linalg.cholesky(corr_mat)

    for _ in range(nstep):
        pars_trial=pars+L@np.random.randn(npar)
        chi_trial=chifun(data,np.insert(pars_trial,3,tau))
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
    return np.array(chain) ,np.array(chivec)


if False:
    curvature = np.load("output/newton_curvature.npy")
    pars = np.load("output/newton_params.npy")
    pars = np.delete(pars,3)
    wmap=ct.read_wmap()
    data=wmap[:,0:3]
    chain,chivec=run_mcmc(pars,data,curvature,nstep=1000, time_out=60*60)

    plt.clf()
    plt.plot(chivec)
    plt.savefig("output/chivec1_t.png")
    plt.clf()
    plt.plot(chain[:,0])
    plt.savefig("output/chain1_0_t.png")
    np.save("output/chain1_t", chain)
    np.save("output/chivec1_t", chivec)

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
    plt.savefig("output/chivec2_t.png")
    plt.clf()
    plt.plot(chain2[:,0])
    plt.savefig("output/chain2_0_t.png")
    np.save("output/chain2_t", chain2)
    np.save("output/chivec2_t", chivec2)
else:
    chain2 = np.load("output/chain2_t.npy")
print("from new chain:")
print("Best params are now: ", np.insert(np.mean(chain2, axis = 0),3, 0.0544))
print("with lower bound for 65/100 confidence: ", np.insert(np.sort(chain2, axis=0)[int(chain2.shape[0]*(1- 0.65)),:], 3, 0.0073))
print("with upper bound for 65/100 confidence: ", np.insert(np.sort(chain2, axis=0)[int(chain2.shape[0]*(0.65)),:], 3, 0.0073))



