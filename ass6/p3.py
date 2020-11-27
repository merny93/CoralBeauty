import numpy as np
import sys
sys.stdout = open('outputs/ratio.txt', 'w')

N=1000000

uv = np.random.rand(N,2)#generate the uv space
uv[:,1] = uv[:,1] *0.5 #shrink the space in v as this is all that is needed to span. I think we can go marginally smaller too but this seems to be reasonable
args = np.argwhere(uv[:,0] < np.exp(-uv[:,1]/uv[:,0]))#find the accepted spots
my_exp = uv[args,1]/uv[args,0]#generate the output

#clean it up
my_exp = my_exp[my_exp < 25]


#just plotting now
import matplotlib.pyplot as plt
plt.hist(my_exp, bins=100, alpha=0.5, label="rustic")
npexp= np.random.exponential(scale= 0.5,size = my_exp.size)
npexp = npexp[(npexp>-25) & (npexp<25)]
plt.hist(npexp, bins=100, alpha=1, label="made by numpy")
plt.title("Ratio of uniforms Exponential random variable")
plt.legend()
plt.savefig("outputs/rat_exp.png")
print("acceptence rate: ", args.size/N)