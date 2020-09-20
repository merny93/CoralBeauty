import numpy as np


#funny story i wrote a exponential version the other day as practice rip should have known to do the log_2
#lets just say log_2 = log for notation 

#log(xy) = log(x) + log(y) and log(a) is reallly easy and fast to evaluate if a is a power of 2 lel so lets just keep dividing by 2 untill we hit number less than 2

#to do this efficiently lets do some bitwise operations

def get_max_base_2(num): #num is a default float (so probably float-64 unless u live in the 1980s)
    int_num = int(num)
    bit_rep = bin(int_num).split("b")[-1] ## will be stupidly slow :) but if written in c will be very fast! (cause this is the real data type)
    max_power =len(bit_rep) -1
    return max_power

##will return the largest power of 2 we can fit in it as well as the remaining bit of num
def rep_base_2(num):
    max_power = get_max_base_2(num)
    divisor  = 1<< max_power
    remainder = num/divisor
    return max_power, remainder


##cool so now we just need to evaluate log_2(x) between 1 and 2 which is arguably a lot lot easier


#based around jons
#and when i say based i mean copy pasted :)
def cheb_fit(fun,ord):
    x=np.linspace(-1,1,ord+1) 
    y=fun(1.5 +x/2)
    mat=np.zeros([ord+1,ord+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(1,ord):
        mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    coeffs=np.linalg.inv(mat)@y
    return coeffs

# coeffs = cheb_fit(np.log2, 10) #to init the coefficitions

def my_log(x, coeffs): #im 2 years old so i find that name funny
    assert(x>0)
    power, to_eval = rep_base_2(x)
    ##the problem is we need to rescale log(x) = log(1.5 + y/2) where since x is between 1 and 2, y is between -1 and 1
    the_log = np.polyval(coeffs[::-1], (to_eval-1.5)*2)
    return power + the_log


coeffs = cheb_fit(np.log2, 31)
x = np.linspace(1,100, 200)
y = np.zeros_like(x)
for i in range(x.size):
    y[i]= my_log(x[i],coeffs)

import matplotlib.pyplot as plt

plt.plot(x,y , label = "my log")
plt.plot(x, np.log2(x), label = "np log")
plt.legend()
plt.show()