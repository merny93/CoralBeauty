##this one might be a little frustrating cause C 
use_c = False

import numpy as np
from matplotlib import pyplot as plt
import ctypes

import sys
sys.stdout = open('outputs/c_random.txt', 'w')

if use_c:
    ##currently doesnt work
    mylib=ctypes.cdll.LoadLibrary("libc.dylib")
    rand=mylib.rand
    rand.argtypes=[]
    rand.restype=ctypes.c_int
else:
    print("my c compiler doesnt want to work defaulting to jon data")
    vals = np.loadtxt("c_data.txt")


###THIS FIRST ATTEMPT DID NOT WORK :(
# 
# vals = np.random.rand(30000,3)
##lets show how bad it is by fitting planes through random sets of 3 points and then plotting this curve

##this did not work
# N=10000
# to_fit = np.random.choice(vals.shape[0],size=3*N, replace=False)

# vals_permute = vals[to_fit,:]
# vals_reshape = vals_permute.reshape(-1,3,3)
# a = vals_reshape[:,1,:] - vals_reshape[:,0,:]
# b = vals_reshape[:,2,:] - vals_reshape[:,0,:]
# normal = np.cross(a,b)
# normal = normal/np.linalg.norm(normal, axis=1)[:,np.newaxis]
# x_dir = normal[:,1]/normal[:,0]
# y_dir = normal[:,2]/normal[:,0]
# print(normal.shape)



# plt.scatter(x_dir,y_dir)
# plt.show()

####ATEMPT NUMERO 2
##ok lets do a mcmc simulation to fit planes to the data!
##this is a lazy mcmc and it runs terribly but it gets the point through!
#this is an mcmc that runs at "zero tmeperature" only accepting steps in the right direction 
#i have also not really made much effort to get a good guess of what the params should be 
#in addition a newthons method sovler probably would have worked here but i love the chaos that comes from MCMC
#so this is the reason i went down this road :)
#
#Also this mcmc runs really poorly mostly becuase of the zero temperature thing and it gets stuck in every local minimum it can find 
#it would benifit from higher temperature and a good estimate for the step

#define a function to get back the number of points

def count_plane(points, params,dv):
    logicall_condition_plus = points[:,2] > (points[:,0]*params[0] + points[:,1]*params[1] + params[2] -dv)
    logicall_condition_minus = points[:,2] < (points[:,0]*params[0] + points[:,1]*params[1] + params[2] +dv)
    return np.count_nonzero(np.logical_and(logicall_condition_plus, logicall_condition_minus))

def run_chain(points, params=[0,0,1e7], dv=1000000, step=[0.05, 0.05,1000000], niter=3000):
    count = count_plane(points, params, dv)
    for _ in range(niter):
        trial_params = np.random.randn(3)*step + params
        new_count = count_plane(points, trial_params, dv)
        if new_count> count:
            params = trial_params
            count = new_count
    return params, count

#run the initial chain
init_params= [-1.3e-1, 7.6e-2, 1.3e7] #i have run this before so i know what the ~good~ numbers are
params, init_count = run_chain(vals, params=init_params)
print("for C random numbers to be ~decent~ we at least expect the points in a given plane to be around: ", vals.shape[0]*0.01 *2) #this is just the fraction of space taken up by the plane
print("the plane fit total for C is: ", init_count)


##jon kindly told us that there is ~30 planes so the next one should be about 1e8/30 away
#so copy pasting the above code for another fit

params2 = params + np.array([0,0,(1e8)/30])
dv = 1000000

##pervent the second chain from messing with the slopes
params2,_ = run_chain(vals, params =params2, step=[0,0,dv/2])

##critically the spacing is now known to be:
spacing=(params2[2] - params[2])
# print("spacing of the planes: ",spacing)

if False:
    ##plot the two inital planes
    xx,yy = np.meshgrid(range(1,int(1e8), int(1e4)),range(1,int(1e8), int(1e4)))
    z = params[0]*xx + params[1]*yy + params[2]
    zz = params2[0]*xx + params2[1]*yy + params2[2]
    # plot the surface

    # Create the figure
    fig = plt.figure()

    # Add an axes
    ax = fig.add_subplot(111,projection='3d')

    # plot the surface
    ax.plot_surface(xx, yy, z, alpha=0.2, color="green")
    ax.plot_surface(xx, yy, zz, alpha=0.2, color="green")
    # and plot the point 

    ax.scatter(vals[:,0], vals[:,1], vals[:,2], s = 0.1)
    plt.show()


    ##done with plotting!
##now lets see what fraction of the points lie on these planes
planes = [params + [0,0, x * spacing] for x in range(-12,30)]
total_count = 0
for plane in planes:
    _, addedvalue= run_chain(vals, plane, step=[0,0, dv/20], niter=500)
    total_count = total_count + addedvalue

print(total_count/vals.shape[0]*100, "percent of the points fall on these ~40 planes")


##Now sadly C doesnt want to play ball on my computer, I have spent a stupid amount of time working on it god damn
# 
# This will obviously not work with python but i guess we can show that :)
# 
# Also im not cheeting since the DV im using is 1 percent of the total space. So with 42 planes which im using to fit we should expect 0.01*42 = 0.42 to be in the planes 
#but this is very much not the case as we have ~all the points in these planes
#also dont go crying about the whole its more than 100 percent! this is a byproduct of my lazy fitting technique
#some points get double counted 


#alright python comparion


python_points = np.random.rand(10000,3)

params, count = run_chain(python_points, params=[0,0,0.3], dv = 0.01, step = [0.1,0.1, 0.1])
#print the count
print("for python random numbers to be ~decent~ we at least expect the following number to be around: ", python_points.shape[0]*0.01 *2)
print("the plane fit total for python is: ", count)


##so interestingly enough python doesnt look perfect (i can also admit that i made a mistake somewhere)
##but my probability knowldge tells me that this is possible since we are optimizing to maximize the number of points in a plane
##hence we sought out that one unluky bit of space that has slightly higher density. 
##interestingly this is a bernouli distribution (either the point falls in the plane or not with probability DV/V_total)
##and i believe the max of a bernoli goes like log(n) or something like that. Curious ay :)
##in any case i think we did enough work to show that this is cool