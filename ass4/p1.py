##i was having fun naming all this things my_something 
##at first i did this with a convolution in real space but i have since realized that i dont think this is what we wanted
# so fft_shit will do the shift using convolutions in fourier space
# also it does a convolution with a delta funciton which ends up shifting.


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def my_sum(my_array):
    return np.sum(my_array)

def my_calloc(my_array):
    return np.zeros_like(my_array)

def my_shift(my_array, my_shift):
    my_lenght = len(my_array)
    my_conv = my_calloc(my_array)
    my_ans = my_calloc(my_array)
    for my_counter in range(my_lenght):
        #tehcnically its:
        #my_ans[i] = np.sum(my_array*my_conv[shift by i])
        my_conv[(my_shift + my_counter)%my_lenght]  = 1 #kinda cheating cause i know its the only 1 so no need to flip but this is the idea of a convolution
        my_ans[my_counter] = my_sum(my_array*my_conv)
        my_conv[(my_shift + my_counter)%my_lenght] = 0
    return my_ans

def fft_shift(arr, shift):
    ##start by defining the array to convolve with:
    conv = np.zeros_like(arr)
    conv[shift] = 1
    ## transform multiply and irfft
    return np.fft.irfft(np.fft.rfft(arr) * np.fft.rfft(conv))

def my_gauss_window(my_size, my_sigma):
    return signal.gaussian(my_size, std=my_sigma)

if __name__ == "__main__":
    my_gauss = my_gauss_window(50, 5)

    my_gauss_shift = (fft_shift(my_gauss, 15))

    plt.clf()
    plt.plot(my_gauss_shift)
    plt.savefig("output/my_shifted_gauss.png")

# yep this looks as expected since if we convolve a gaussian with itself its gonna look "the same " when you are not shifting 
# then as u "slide" it starts to be more and more different untill the circular nature takes back over and we match on the other side
# not surprises here