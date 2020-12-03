import numpy as np

"""
for part a as we know that the value of the potential at (1,0) is the average of its enigbors and we know the value of the neigbors 
We can find out what it is at zero

"""

# V_10 = (V_00 + V_11 + V_20 + V_1-1)/4
# so V_00 = V_10 * 4 - (V_11 + V_20 + V_1-1)
#or in math 

V_cent_offset = np.log(1)*4 -(np.log(np.sqrt(2)) + np.log(2) + np.log(np.sqrt(2)))

#so to set the potential to 1 at zero we have to divide by that


print(["pot at " + str(x) + " is: " + str(np.log(x)/V_cent_offset) for x in [1,2,5]])




##lets do the bonus to learn something. We will generate the grid and generate the potential in it 
##so i failed miserably at the bonus and i recomend u skip to line 100 where i do part b
##stealing some code from myself which is a complexd way to generate a mesh grid 
N = 1000
template = np.zeros((N,N), dtype=float)
idx = np.roll(np.arange(-template.shape[0]//2, template.shape[0]//2, dtype=float), N//2)
basis_vec = np.array([idx for _ in range(2)])
dist_grid = np.array(np.meshgrid(*basis_vec))

template = np.log(np.linalg.norm(dist_grid, axis=0, keepdims=False))/V_cent_offset
template[0,0] = 1
from matplotlib import pyplot as plt

def convolve(ar1,ar2):
    assert(ar1.shape == ar2.shape)
    return np.fft.irfft2(np.fft.rfft2(ar1) * np.fft.rfft2(ar2))

#now we need to compute the laplacian
def get_charge(pot):
    opperator = np.zeros_like(pot)
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    opperator[:2,:2] = lap[1:,1:]
    opperator[-1,-1] = lap[0,0]
    opperator[-1,:2] = lap[0,1:]
    opperator[:2,-1] = lap[1:,0]
    return convolve(pot, opperator)

def update_pot(pot, real):
    charge = get_charge(pot)
    # charge[:,charge.shape[1]//2 - 1: charge.shape[1]//2 + 1] = 0
    # charge[charge.shape[1]//2 - 1: charge.shape[1]//2 + 1, :] = 0
    charge = charge - real
    charge_pot = convolve(pot,charge)
    # plt.imshow(charge)
    # plt.show()
    # plt.imshow(charge_pot)
    # plt.show()
    new_pot = pot - charge_pot
    new_pot = new_pot/new_pot[0,0]
    return new_pot


true_charge = np.zeros_like(template)
true_charge[0,0] = 1



pot_copy = convolve(true_charge,template)


# plt.imshow(pot_copy)
# plt.show()
for _ in range(2):
    pot_copy = update_pot(pot_copy, true_charge)
    # plt.imshow(pot_copy)
    # plt.show()
new_pot = pot_copy

# plt.imshow(new_pot)
# plt.show()

















## having officially given up on the bonus 
## the matrix is gonna be toeplitz. Can draw it out to see as along the diagonal we have strength at distance zero and going out either way it drops off by 1 at a time
#but we need not solve it everywhere. We need but to solve it on a line. Saying that the l is 100 then we find G to be

N_l = 100
from scipy.linalg import toeplitz
G = toeplitz(np.ravel(template)[:N_l], np.ravel(template)[:N_l])

### our bar is simply 
bar_pot = np.ones(N_l)

##and we know that G*rho = bar_pot 
#we can solve this with a simple matrix inversion but i hear that is no fun


def conjgrad(A,b,x):
    r = b - A@x
    p = r
    rsold = r@r
    for _ in range(b.size):
        Ap = A@p
        alpha = rsold/(p@Ap)
        x = x+ (alpha * p)
        r = r - (alpha * Ap)
        rsnew = r@r
        if np.sqrt(rsnew) < 1e-8:
            return x
        p = r + (rsnew/rsold) *p
        rsold = rsnew
    return x

# a = np.array([[3,1], [1,2]])
# b = np.array([9,8])
# x = np.linalg.solve(a, b)
# x_con = conjgrad(a,b, np.ones(2))

# print(x)
# print(x_con)

## seems to work so thats nice 

##so more to the point

rho = conjgrad(G, bar_pot, np.ones(N_l))

plt.clf()
plt.plot(rho)
plt.title("Charge on the Bar")
plt.savefig("charge_on_bar.png")

##this is but a matter of convolving the potential with the box im not gonna bother with edge effects oh well

box_mask = np.zeros_like(template)
box_mask[N//2 - N_l//2: N//2 + N_l//2, N//2-N_l//2] = rho
box_mask[N//2 - N_l//2: N//2 + N_l//2, N//2+N_l//2] = rho
box_mask[N//2-N_l//2, N//2 - N_l//2: N//2 + N_l//2] = rho
box_mask[N//2+N_l//2, N//2 - N_l//2: N//2 + N_l//2] = rho

final = convolve(box_mask, template)
plt.clf()
plt.imshow(final)
plt.savefig("pot_from_box.png")

##now to take the derivatives
xx,yy = np.meshgrid(np.arange(N_l), np.arange(N_l))

u = np.roll(final,1, axis = 1) - np.roll(final,-1, axis = 1) 
v = np.roll(final,1, axis = 0) - np.roll(final,-1, axis = 0) 

plt.clf()
plt.quiver(xx,yy,u[N//2: N//2+N_l, N//2 - N_l//2: N//2 + N_l//2],v[N//2: N//2+N_l, N//2 - N_l//2: N//2 + N_l//2])
plt.title("Electric Field")
plt.savefig("Field near bound")
##seems to follow the lore

