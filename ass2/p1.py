import numpy as np #our lord and savior



##function to try it on (a very good candiadate as poles and what not)
def lorentz(x, return_num =False):
    lorentz.counter += x.size
    if return_num:
        return 1/(1+x**2), lorentz.counter
    return 1/(1+x**2)
lorentz.counter= 0 #init counter

##copy paste from class to compare
def integrate_step(fun,x1,x2,tol):
    #print('integrating from ',x1,' to ',x2)
    x=np.linspace(x1,x2,5)
    y=fun(x)
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,x1,xm,tol/2)
        a2=integrate_step(fun,xm,x2,tol/2)
        return a1+a2

left = -1
right = 1
tol = 1e-6
lorentz.counter= 0 #init counter
ans = integrate_step(lorentz,left,right, tol)
print("answer from easy version:", ans, "with:", lorentz.counter, "function evaluations")

def integrate_from_prev(fun,x1,x2,tol, prev = None):
    #print('integrating from ',x1,' to ',x2)
    if prev is None:
        x=np.linspace(x1,x2,5)
        y=fun(x)
    else: #we got our prior!!!
        x=np.linspace(x1,x2,5)[1:4:2] #we are just missing the extra 2 points
        y_missing = fun(x) #also i love that jon uses fun and not func, makes coding more fun (pun intended)
        y = np.zeros(5)
        y[1:4:2] = y_missing
        y[::2] = prev
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_from_prev(fun,x1,xm,tol/2, prev = y[0:3])
        a2=integrate_from_prev(fun,xm,x2,tol/2, prev = y[2:])
        return a1+a2

lorentz.counter= 0 #init counter
ans = integrate_from_prev(lorentz,left,right, tol)
print("answer from saving version:", ans, "with:", lorentz.counter, "function evaluations")
##about a factor of 2 as expected


##idk if my implementation here actually saves much time:
# a better implementation would pass memory addresses along to avoid extra malloc and free calls in the undelying c code
# wayyy out of the scope of the class tho lel