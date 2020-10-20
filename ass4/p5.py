import numpy as np

#done with the antics this will be normal code

## lets fft a non integer sine wave 
x = np.arange(120)
my_sin = np.sin(2 * np.pi * 0.123 * x)

my_ft = np.fft.fft(my_sin)

from matplotlib import pyplot as plt
plt.clf()
plt.plot(np.abs(my_ft), "*")
plt.savefig("output/spectral_leackage.png")

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