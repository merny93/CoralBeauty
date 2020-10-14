import numpy as my_np

def my_conjugate(my_array):
    return my_np.conjugate(my_array)

def my_fft(my_array):
    return my_np.fft.fft(my_array)

def my_inverse_fft(my_array):
    return my_np.fft.ifft(my_array)

def my_correlation(my_first_array, my_second_array):
    assert(len(my_first_array) == len(my_second_array))
    my_first_array_discrete_fourier_transform = my_fft(my_first_array)
    my_second_array_discrete_fourier_transform = my_fft(my_second_array)
    my_correlation_fourier_space_representation = my_first_array_discrete_fourier_transform * my_conjugate(my_second_array_discrete_fourier_transform)
    my_correlated_array = my_inverse_fft(my_correlation_fourier_space_representation)
    return my_correlated_array

import matplotlib.pyplot as plt
from scipy import signal
def my_gauss_window(my_size, my_sigma):
    return signal.gaussian(my_size, std=my_sigma)

if __name__ == "__main__":
    my_gauss = my_gauss_window(50, 5)

    my_gauss_self_correlation = my_correlation(my_gauss, my_gauss)

    plt.clf()
    plt.plot(my_gauss_self_correlation)
    plt.savefig("output/my_guass_self_correlation")