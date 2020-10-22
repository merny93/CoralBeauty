import numpy as np

#done with the antics this will be normal code

## lets fft a non integer sine wave 
x = np.arange(111)
l = 0.123 * x.size
my_sin = np.sin(2 * np.pi * l * x / x.size)

my_ft = np.fft.fft(my_sin)

from matplotlib import pyplot as plt


##and we compare to analytics

def analytic(k,l):
    comp_exp = lambda arg: np.exp(-2j* np.pi * arg)
    term_1 = (1-comp_exp(k-l))/((1-comp_exp((k-l)/(x.size))))
    term_2 = (1-comp_exp(k+l))/((1-comp_exp((k+l)/(x.size))))
    return 1/(2j) * (term_1 - term_2)

expected_ft = analytic(x, l)

print("the difference between computed and expected: ", np.std((my_ft) - (expected_ft)))
# print((my_ft) - (expected_ft))
plt.clf()
plt.plot(np.abs(expected_ft), "*", label="expected")
plt.plot(np.abs(my_ft), '*', label="computed")
plt.legend()
plt.savefig("output/spectral_leackage.png")
plt.clf()
plt.plot(np.abs(expected_ft - my_ft), "*")
plt.savefig("output/ratio_analtocomp.png")
##idk why the hell they arent identical. This is some numpy sneaking bullshit



##now we multiply by 0.5  - 0.5 cos(...) which numpy has packaged for us as the hanning window

my_sin_window = my_sin * np.hanning(my_sin.size)

#now fft and lets see what happened 

ft_windowed = np.fft.fft(my_sin_window)

plt.clf()
plt.plot(np.abs(ft_windowed), "*" , label="windowed")
plt.plot(np.abs(my_ft), "*", label="original")
plt.legend()
plt.savefig("output/windowed_ft.png")

#now to turn my_ft into ft_windowed with a linear operation

## the return of the king
from scipy.linalg import toeplitz

##create the "convolution" matrix
##this is a mostly diagonal matrix which mixes neigbors
# mostly since it wrapes around so the corners have values too
# divide by N since my_ft also got multiplied by N so to keep things unity
constructor = np.zeros_like(my_ft)
constructor[0] = (x.size /2)/x.size
constructor[1] = -(x.size / 4)/x.size
constructor[-1] = -(x.size /4)/x.size
wind_mat = toeplitz(constructor, constructor)

rec_win_ft = np.dot(wind_mat, my_ft)

plt.clf()
plt.plot(np.abs(rec_win_ft), "*", label="reconstructed")
plt.plot(np.abs(ft_windowed), "*" ,markersize=3, label="windowed")
plt.plot(np.abs(my_ft), "*", label="original")
plt.legend()
plt.savefig("output/rec_windowed_ft.png")

## see they are identical 