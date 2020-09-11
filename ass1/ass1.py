####Hello ladies and gentelmen welcome to the shitshow that is my assignement:
##also the repo name is a surprise for jon :)


#### Consider the derivative as folows f'(x) = [8f(x+dx) - 8f(x-dx) - f(x+2dx) + f(x - 2dx)]/12dx
#this will magically remove the second order problem too :)

#as for the float precision. We only care about order of magnitude since anyways we are gonna make order of magnitude assumptions about the function vals
#the error term is gonna be f''''(x)dx^3 so f' = () + dx^3 f'''' + g \epsilon f /dx
#taking the partial with dx we get f''''dx^2 - g \epsilon f/dx^2
##s so dx = \sqrt[4]{\frac{g\epsilon f}{f''''}}

##coolcoolcool nodoubt nodoubt nodoubt 

#conviniently if $f(x) = e^x$ we have that $f = f' = f'' = \cdots $ and at around unity we can set it all to 1 :)

import numpy as np

dx = np.power(1e-16, 0.25) ## create the dx for the unity case. it will be different if we have exp(0.001x) cause the derivatives get tinny
x = np.arange(-0.5,0.5, dx) ##get the xs seperated properly 
y = np.exp(x)

der_mat = np.zeros((x.size - 3, x.size)) ##create a full mat representation of the derigvative opperator
#this is obviously slow cause its a sparse band diagonal toeplitz matrix but python. Maybe i wont be lazy and will call something better than np.dot but probs not

from scipy.linalg import toeplitz
##create a full mat representation of the derigvative opperator
#this is obviously slow cause its a sparse band diagonal toeplitz matrix but python. Maybe i wont be lazy and will call something better than np.dot but probs not
der_op = np.array([1,-8, 0, 8,-1])
der_op = der_op/(12*dx)
der_row = np.zeros(x.size)
der_row[0:5] = der_op
der_col = np.zeros(x.size - 4)
der_col[0] = der_op[0]
der_mat = toeplitz(der_col,der_row)

y_dir = np.dot(der_mat, y.T)
import matplotlib.pyplot as plt

# plt.plot(y_dir)
# plt.plot(y[2:-2])
# plt.show()

print("accuracy is for e^x: ", np.std((y_dir-y[2:-2]))) 
###---------------------------------------------------------------------------##############





##for the next one imma copy paste some code and only change the function and the dx (this is a good example for why functions are great in programing)

##no if f=exp(0.01x) then f''''(x) = (0.01)^4exp(0.01) ~ 1e-8 so lets incorporate that

dx = np.power(1e-16/1e-8, 0.25) ## create the dx for the unity case. it will be different if we have exp(0.001x) cause the derivatives get tinny
x = np.arange(-0.5,0.5, dx) ##get the xs seperated properly 
y = np.exp(0.01 * x)
y_dir_real = 0.01 *np.exp(0.01 * x)
der_mat = np.zeros((x.size - 3, x.size)) ##create a full mat representation of the derigvative opperator
#this is obviously slow cause its a sparse band diagonal toeplitz matrix but python. Maybe i wont be lazy and will call something better than np.dot but probs not

from scipy.linalg import toeplitz
##create a full mat representation of the derigvative opperator
#this is obviously slow cause its a sparse band diagonal toeplitz matrix but python. Maybe i wont be lazy and will call something better than np.dot but probs not
der_op = np.array([1,-8, 0, 8,-1])
der_op = der_op/(12*dx)
der_row = np.zeros(x.size)
der_row[0:5] = der_op
der_col = np.zeros(x.size - 4)
der_col[0] = der_op[0]
der_mat = toeplitz(der_col,der_row)

y_dir = np.dot(der_mat, y.T)
import matplotlib.pyplot as plt

# plt.plot(y_dir)
# plt.plot(y_dir_real[2:-2])
# plt.show()

print("accuracy is for e^0.01x: ", np.std((y_dir-y_dir_real[2:-2]))) 


###########NEXT PROBLEM 

##seems like the first col is temp in kelvin and second col is voltage 
vol = []
temp = []
with open("lakeshore.txt", "r") as data:
    while True:
        try:
            row = data.readline().split("\t")
            temp_t = float(row[0])
            volt_t = float(row[1])
            vol.append(volt_t)
            temp.append(temp_t)
        except:
            break

#switch to c style data structure
vol = np.array(vol) 
temp = np.array(temp) #dont worry im just as terrified of using temp as a non temporary array 
from scipy import interpolate ## get the magic library lel

f_full = interpolate.interp1d(vol, temp, kind= "cubic")

x_fine = np.linspace(vol[0],vol[-1],num = 5000)

###idk it wasnt working whatever cubic is perfectly fine lmao
##we get temp_full = interpolate.splev(x_fine, f_full)
#temp_full = interpolate.splev(x_fine, f_full)


temp_full = f_full(x_fine)

# plt.plot(vol,temp, "*")
# plt.plot(x_fine, temp_full)
# plt.show()

#to get a rough estimat lets pretend we have a portion of the data we have, re do the above and see how well it fits the points we put asside :)

test_ind = np.random.permutation(list(range(vol.size)[5:-5]))[0:int(vol.size / 5)]
keep_ind = [x for x in range(vol.size) if x not in test_ind]
vol_cut = vol[keep_ind]
temp_cut = temp[keep_ind]

f_full_test = interpolate.interp1d(vol_cut, temp_cut, kind= "cubic")

vol_test = vol[test_ind]
temp_test = temp[test_ind]

temp_m = f_full_test(vol_test)
print("the maxxx error is :", np.std(temp_m-temp_test))

##as time goes on im less and less interested in adding comments lel
N = 6 #chose an even power to give the polynomial a fighting chance lol
x_c = np.linspace(-np.pi/2, np.pi/2, num=N-1)
y_c = np.cos(x_c) +0.5
x_f = np.linspace(-np.pi/2, np.pi/2, num=1000)
y_f = np.cos(x_f) +0.5 




pol = np.polyfit(x_c,y_c, N)
y_pol = np.polyval(pol, x_f)

##a wise man once told me to not re-write code written by smarter people than me so hee it goes
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q, mat

nom = 1
p,q, mat = rat_fit(x_c, y_c, nom,N-nom)
y_rat = rat_eval(p,q, x_f)
print(p,q)
spl = interpolate.splrep(x_c, y_c)
y_spl = interpolate.splev(x_f, spl)


# import scipy.linalg as la
# print(mat)
# w, v = np.linalg.eig(mat[:,:])
# print(w[-1], v[-1])
# print(la.svdvals(mat))

plt.plot(x_c,y_c, "*")
plt.plot(x_f, y_spl)
plt.plot(x_f, y_rat)
plt.plot(x_f, y_pol)
plt.plot(x_f, y_f)
plt.show()

print("std of poly:", np.std(y_f-y_pol))
# print("std of spline:", np.std(y_f-y_spl))
print("std of rat:", np.std(y_f-y_rat))

x_c = np.linspace(-1, 1, num=N-1)
y_c = np.cos(x_c)
x_f = np.linspace(-1, 1, num=1000)
y_f = np.cos(x_f)

def lorentzian( x, a, b, c ):
    return a  / ( 1 + b * x**2) + c

from scipy.optimize import curve_fit

popt, _ = curve_fit(lorentzian, x_c, y_c)

y_lor = lorentzian(x_f, *popt)

plt.plot(x_f,y_f)
plt.plot(x_f, y_lor)
plt.show()
print("std of lor:", np.std(y_f-y_lor))


##next problem

##symetry argument about the field being radially out yadadada this isnt an e&m class
