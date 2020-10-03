import numpy as np
from matplotlib import pyplot as plt

vals = []
with open("sky_data.txt", "r") as fl:
    while ((line := fl.readline())[0] != "/"): ##guess who installed python3.8
        if line[0] == "#":
            continue
        vals.append([float(x) for x in line.split()])
vals = np.array(vals)
#print(vals[-1,0])


from sky_exmp import get_spectrum 
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=vals

#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
# plt.plot(wmap[:,0],wmap[:,1],'.')

cmb=get_spectrum(pars)
# plt.plot(cmb)
# plt.show()

##ok so that works 
def fit_wrap(x, pars):
    return get_spectrum(pars)[2:x.shape[0]+2]

def my_chi(wmap, pars, func = fit_wrap):
    return np.sum((fit_wrap(wmap[:,0],pars) - wmap[:,1])**2 / wmap[:,2]**2)

print("Chi^2 magic is: ", my_chi(wmap, np.array([6.93261622e1, 2.24909219e-2, 1.13912680e-1, 5.00000000e-2, 2.04249787e-9, 9.69760318e-1]), fit_wrap))
print("Chi^2 from jon params is: ", chi_org := my_chi(wmap, pars, fit_wrap))


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



Ninv= np.diag(1/(wmap[:,2]**2))
y = wmap[:,1]
x = wmap[:,0]
dpar = pars*1e-2 

if False:
    print("Now we do a Newthons method solver")
    for i in range(10):
        model=fit_wrap(x,pars)
        derivs=num_deriv(fit_wrap,x,pars,dpar)
        derivs = np.delete(derivs, 3,1)
        
        resid=y-model
        lhs=derivs.T@Ninv@derivs
        rhs=derivs.T@Ninv@resid
        lhs_inv=np.linalg.inv(lhs)
        step=lhs_inv@rhs
        
        step = np.insert(step, 3, 0)
        pars=pars+step
        ##so that tau never changes
        dpar = np.insert(np.sqrt(np.diag(lhs_inv))*0.1, 3, dpar[3])
        print("On the: " , i, " itteration and params are: ", pars)
        print("Chi^2 is now: ", chi_now := my_chi(wmap, pars, fit_wrap) )
        if np.abs(chi_now - chi_org) < 0.1:
            print("we have converged")
            break
        chi_org = chi_now


chi_sqrd = np.sum((fit_wrap(x,pars) - wmap[:,1])**2 / wmap[:,2]**2)
print("New chi^2 after the newthon method is :" , chi_sqrd)

##i Guess we can plot a graph 

def run_mcmc(pars,data,par_step,chifun,nstep=500):
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    chi_cur=chifun(data,pars)
    for i in range(nstep):
        pars_trial=pars+np.random.randn(npar)*par_step
        if pars_trial[3] < 0:
            pars_trial = pars[3]
        chi_trial=chifun(data,pars_trial)
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            print("accepted")
            pars=pars_trial
            chi_cur=chi_trial
        chain[i,:]=pars
        chivec[i]=chi_cur
    return chain,chivec

##next problem
pars = np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
par_sigs=pars*0.2
data=wmap[:,0:3]
chain,chivec=run_mcmc(pars,data,par_sigs,my_chi,nstep=200)
par_sigs=np.std(chain,axis=0)
par_means=np.mean(chain,axis=0)
plt.plot(chivec)
plt.title("Chi^2 over the steps")
plt.savefig("Chidecent.png")

plt.plot(chain[:,3])
plt.title("tau steps")
plt.savefig("tausteps.png")
print(par_means, par_sigs, np.mean(chivec), np.min(chivec))

nchain=4
all_chains=[None]*nchain
for i in range(nchain):
    pars_start=par_means+3*par_sigs*np.random.randn(len(pars))
    chain,chivec=run_mcmc(pars_start,data,par_sigs,my_chi,nstep=250)
    all_chains[i]=chain


np.save("allchains.npz", np.array(all_chains))


for i in range(nchain):
    for j in range(i+1,nchain):
        mean1=np.mean(all_chains[i],axis=0)
        mean2=np.mean(all_chains[j],axis=0)
        std1=np.std(all_chains[i],axis=0)
        std2=np.std(all_chains[j],axis=0)
        print('param difference in sigma is ',(mean1-mean2)/(0.5*(std1+std2)))