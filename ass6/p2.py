##start by stating that we can not use a gaussian to overbound an exponential 
##this is since
##gaussian goes like e^-x^2
##exponential goes like e^-x
##so gaussian shrinks faster and thus no matter what constant we use we will be in deep shite 
## the solution is to use a lorentzian to see that it is always bigger we take the ratio as follows:
## e^-x/(1/(1+x^2)) = (1+x^2)e^-x = (1+x^2)/e^x ~~ (1 + x^2)/(1 + x + x^2/2 + x^3/6 \cdot) < 1 
##yeah i was a touch lazy there and techincally we need a factor of 2 but with a bit more care we dont even need a factor of 2

##actually i guess i was hasty. We technically can use a gaussian but its simply a really bad idea.
##assuming we cut off the tail somewhere we can find a constant thats enough bigger to make sure its biggere everywhere
##yet this is an absolutely terrible idea dont do this as at the tail efficiency will be decent but everywhere else it will be trash 

#Power law will work too with basically any power by the same proof as the lorentzian (except there is no 1 which doesnt change much)
#im a fan of the lorentzian so i might as well use that and np.random.standard_cauchy() already packages it so thats nice 


##alright now lets code this bad boi up
import numpy as np

import sys
sys.stdout = open('outputs/rejection.txt', 'w')

##im gonan use np.random.standard_cauchy but its just an tan so i guess not much to do 
def my_lor(N):
    return np.tan((np.random.rand(N)-0.5)*np.pi)

def lor_func(x):
    return 1/(1 * (1 + x**2))

def exp_func(x):
    return np.exp(-x)

import matplotlib.pyplot as plt
N = 10000000
s = my_lor(N)
#s = np.random.standard_cauchy(1000000)
s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
plt.hist(s, bins=100)
plt.title("sample cauchy random variable")
plt.savefig("outputs/cauchy.png")
plt.clf()
##works cool beens


##ok for efficiency sake lets do the absolute value of the lorenzian so its as if we did twice as many
my_bound = np.abs(s)
N = my_bound.size
accept_prob = np.random.rand(N)
my_exp_pos = np.argwhere(exp_func(my_bound)/lor_func(my_bound) > accept_prob)
my_exp = my_bound[my_exp_pos]
accept_rate = my_exp_pos.size/N
plt.hist(my_exp, bins=100, alpha=0.5, label="rustic")
npexp= np.random.exponential(size = my_exp.size)
npexp = npexp[(npexp>-25) & (npexp<25)]
plt.hist(npexp, bins=100, alpha=0.5, label="made by numpy")
plt.title("sample Exponential random variable")
plt.legend()
plt.savefig("outputs/exp.png")

print("acceptance rate was: ", accept_rate)
##playing around with the functions this one seems pretty good as an option 
##that being said we can do piece wise continous to better fit in the closer bits.
