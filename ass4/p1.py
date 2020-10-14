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
        my_conv[(my_shift + my_counter)%my_lenght]  = 1
        my_ans[my_counter] = my_sum(my_array*my_conv)
        my_conv[(my_shift + my_counter)%my_lenght] = 0
    return my_ans

def my_gauss_window(my_size, my_sigma):
    return signal.gaussian(my_size, std=my_sigma)

if __name__ == "__main__":
    my_gauss = my_gauss_window(50, 5)

    my_gauss_shift = (my_shift(my_gauss, 25))

    plt.clf()
    plt.plot(my_gauss_shift)
    plt.savefig("output/my_shifted_gauss.png")