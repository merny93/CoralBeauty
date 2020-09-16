####Hello ladies and gentelmen welcome to the shitshow that is my assignement:
##also the repo name is a surprise for jon :)


#### Consider the derivative as folows f'(x) = [8f(x+dx) - 8f(x-dx) - f(x+2dx) + f(x - 2dx)]/12dx
#this will magically remove the third order problem too :)
##and 4th is dead since 
##expand taylor to see this. I swear it workes 

#as for the float precision. We only care about order of magnitude since anyways we are gonna make order of magnitude assumptions about the function vals
#the error term is gonna be f'''''(x)dx^4 so f' = (something) + dx^4 f''''' + g \epsilon f /dx
#taking the partial with dx we get f'''''dx^3 - g \epsilon f/dx^2
##s so dx = \sqrt[5]{\frac{\epsilon f}{f'''''}}

##coolcoolcool nodoubt nodoubt nodoubt 

#conviniently if $f(x) = e^x$ we have that $f = f' = f'' = \cdots $ and at around unity we can set it all to 1 :)

import numpy as np

dx = np.power(1e-16, 0.2) ## create the dx for the unity case. it will be different if we have exp(0.001x) cause the derivatives get tinny
x = np.arange(-0.5,0.5, dx) ##get the xs seperated properly 
y = np.exp(x)

der_mat = np.zeros((x.size - 3, x.size)) ##create a full mat representation of the derigvative opperator
#this is obviously slow cause its a sparse band diagonal toeplitz matrix but python. Maybe i wont be lazy and will call something better than np.dot but probs not

from scipy.linalg import toeplitz
##create a full mat representation of the derigvative opperator
#this is obviously slow cause its a sparse band diagonal toeplitz matrix but python just matmuls it. Maybe i wont be lazy and will call something better than np.dot but probs not
der_op = np.array([1,-8, 0, 8,-1])
der_op = der_op/(12*dx)
der_row = np.zeros(x.size)
der_row[0:5] = der_op
der_col = np.zeros(x.size - 4)
der_col[0] = der_op[0]
der_mat = toeplitz(der_col,der_row)

y_dir = np.dot(der_mat, y.T)
import matplotlib.pyplot as plt

plt.plot(x[2:-2], y_dir)
plt.plot(x[2:-2],y[2:-2])
plt.title("derivative of e^x")
plt.show()

print("accuracy is for e^x: ", np.std((y_dir-y[2:-2]))) 
###---------------------------------------------------------------------------##############





##for the next one imma copy paste some code and only change the function and the dx (this is a good example for why functions are great in programing)

##no if f=exp(0.01x) then f'''''(x) = (0.01)^5exp(0.01) ~ 1e-10 so lets incorporate that

dx = np.power(1e-16/1e-10, 0.2) ## create the dx for the unity case. it will be different if we have exp(0.001x) cause the derivatives get tinny
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

plt.plot(x[2:-2], y_dir)
plt.plot(x[2:-2], y_dir_real[2:-2])
plt.title("derivative of e^0.01x")
plt.show()

print("accuracy is for e^0.01x: ", np.std((y_dir-y_dir_real[2:-2]))) 


###########NEXT PROBLEM ****************************************************************

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

plt.plot(vol,temp, "*")
plt.plot(x_fine, temp_full)
plt.title("cubic interpolation of temp vs voltage")
plt.show()

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



###NEXT PROBLEM *******************************************************

##as time goes on im less and less interested in adding comments lel
N = 6 #chose an even power to give the polynomial a fighting chance lol
x_c = np.linspace(-np.pi/2, np.pi/2, num=N-1)
y_c = np.cos(x_c) +0.5 ##ah so why the +0.5? good question! 
##so part of this question is why does rat fit take a crap when you try it here. Well the answer is that the y val hits zero at pi/2 and -pi/2. 
##this makes the mat in rat_fit singular and thus the regular inverse doesnt work and pinv gives a psudo inverse which is not great.
##so the solution to this is simply to offset the function by  a bit so it never touches zero. No harm no fowl.
x_f = np.linspace(-np.pi/2, np.pi/2, num=1000)
y_f = np.cos(x_f) +0.5 




##polyfit!
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
    pars=np.dot(np.linalg.pinv(mat),y) ##I switched to pinv cause its just better in every way. Small singular values screw everything up in life.
    p=pars[:n]
    q=pars[n:]
    return p,q, mat

##RAT FIT
#but why simon, Why does the matrix care that it touched zero?
##recall it is defined as mat = [1, x, x^2, ..., x^n, -yx, -yx^2, ... , -yx^m]
##so if we have a zero y we will have a lot of zeros in the matrix (all entries that touch the q vector)
# and thus it will mean that q has one less equation than it needs and hence the matrix is underdetermined or in fancy words it is singular
# pinv will compute the best guess of the inverse by thorwing out the part of the vectorspace coresponding to the null space. (gross simplification)

nom = 1
p,q, mat = rat_fit(x_c, y_c, nom,N-nom)
y_rat = rat_eval(p,q, x_f)
print(p,q)

##spline fit this has same number of points so its fair
spl = interpolate.splrep(x_c, y_c)
y_spl = interpolate.splev(x_f, spl)

#me messing with the svd decomposition to see what the hell is happening
# import scipy.linalg as la
# print(mat)
# w, v = np.linalg.eig(mat[:,:])
# print(w[-1], v[-1])
# print(la.svdvals(mat))

plt.plot(x_c,y_c, "*")
plt.plot(x_f, y_spl, label="spline")
plt.plot(x_f, y_rat, label="rational")
plt.plot(x_f, y_pol, label="polynomial")
plt.plot(x_f, y_f, label="reality")
plt.title("Different fits of cos func")
plt.legend()
plt.show()

print("std of poly:", np.std(y_f-y_pol))
print("std of spline:", np.std(y_f-y_spl))
print("std of rat:", np.std(y_f-y_rat))

###now for the lorezian using curve_fit which is a generalized least squares fitter i beleive
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
plt.title("lorenzien fit of cos (so similar u cant tell)")
plt.show()
print("std of lor:", np.std(y_f-y_lor))
##which works increadibly well

####ohhh turns out i had to fit the stuff to the lorenzian doe.
##i guess we shouldnt be surprised that a lorenzian and a cos is the same cause they are the same lol



N = 6 #chose an even power to give the polynomial a fighting chance lol
x_c = np.linspace(-1, 1, num=N-1)
y_c = lorentzian(x_c ,1,1,0)
x_f = np.linspace(-1, 1, num=1000)
y_f = lorentzian(x_f ,1,1,0)

##polyfit!
pol = np.polyfit(x_c,y_c, N)
y_pol = np.polyval(pol, x_f)


nom = 1
p,q, mat = rat_fit(x_c, y_c, nom,N-nom)
y_rat = rat_eval(p,q, x_f)
print(p,q)

##spline fit this has same number of points so its fair
spl = interpolate.splrep(x_c, y_c)
y_spl = interpolate.splev(x_f, spl)


plt.plot(x_c,y_c, "*")
plt.plot(x_f, y_spl, label="spline")
plt.plot(x_f, y_rat, label="rational")
plt.plot(x_f, y_pol, label="polynomial")
plt.plot(x_f, y_f, label="reality")
plt.title("Different fits of lorezian func")
plt.legend()
plt.show()

print("std of poly:", np.std(y_f-y_pol))
print("std of spline:", np.std(y_f-y_spl))
print("std of rat:", np.std(y_f-y_rat))##fits perfectly cause its a rational funciton....
##next problem **********************************************

##symetry argument about the field being radially out yadadada this isnt an e&m class
#E_z(z) = \frac{\lambda 2 pi R' z}{\epsilon_0 (z^2 + R'^2)^{3/2}}


## if total charge is Q then Q = \int_0^\pi 2 \pi R \lambda \sin(\theta) d\theta = 4 \pi R \lambda but i guess this is kinda besides the point so whatever
lamb = 1 #dont use lambda as that is a keyword in python
e_0 = 1 #also besides the point i know its not actually 1
R = 1 #who cares what we set R to

##now we want to integrate over a sphere at position y wrt to the closest point so R' = sin(y/2R * pi)


##to get a sphere we integrate over rings that form the sphere. where we have the field due to a ring at angle theta given by:
def E_z(theta, z):
    return (lamb * 2 * np.pi * R*np.sin(theta) * (z - R*np.cos(theta))) / np.power((R*np.sin(theta))**2 +   (z - R*np.cos(theta))**2, 3/2)


## obviously singular at R so lets just avoid it
z_vals = np.linspace(0, R*5, num=154) ##just read assignement god damn it 
z_vals = np.append(z_vals, R)



from scipy import integrate

res = [integrate.quad(E_z, 0, np.pi, args=(x))[0] for x in z_vals] ##integrate over the stuffs for outside
res = np.array(res) #magic c structure



plt.scatter(z_vals, res)
plt.title("field integrated by quad")
plt.axvline(x=R)
##wow it looks as expected
plt.show()

##now to stir up some shit let me use my own integrator with expected results. 
## lets be lazy and just straight up call np.sum as a powerful integrator or for fun lets turn it into a linear oppertor
##now im doing the worst possible integration technique (that is zeroth order interpolation)
##Im summing rectangles which has linear error. but i guess everything works out

##first compute E_z at all the different z of interest and all the different theta

N = 1000
theta_vec = np.linspace(0, np.pi, num = N)
z_vec = z_vals

E_z_mat = np.zeros((theta_vec.size, z_vec.size))
theta_mat = np.zeros_like(E_z_mat) 
theta_mat[:]= theta_vec[:,np.newaxis]

z_mat = np.zeros_like(E_z_mat)
z_mat[:] = z_vec[np.newaxis, :]

E_z_mat = E_z(theta_mat, z_mat)

##now the completely stupid linear operation that pertends to be a integral:

int_mat = np.zeros_like(theta_vec)
int_mat[:] = (theta_vec[-1] - theta_vec[0])/theta_vec.size


E_z_int = np.dot(int_mat, E_z_mat) #nans are annoying and my "integrator" obviously doesnt handel them and just gives me trash for that row

plt.scatter(z_vec, E_z_int)
plt.title("field integrated by me")
plt.axvline(x=R)
plt.show()

##lets plot the residuals cause why not

plt.scatter(z_vec, res - E_z_int)
plt.title("residuals")
plt.axvline(x=R)
plt.show()
##as expected it gets bad where the function changes sharply

## excluding the nan
print("the error between my shitty solution and the good one (excluding the time difference):", np.std(res[:-1] - E_z_int[:-1]))

##and now some brain fun
assert('jon' is not None)
assert('None' is (not None))