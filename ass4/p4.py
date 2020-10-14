import numpy as np; from scipy import signal; import matplotlib.pyplot as plt; plt.clf()
no_circ_conv = lambda arr1, arr2 : np.fft.ifft(np.fft.fft(np.pad(arr1, (0,len(arr1)), mode='constant', constant_values=(0,0))) * np.fft.fft(np.pad(arr2, (0,len(arr2)), mode='constant', constant_values=(0,0))))
plt.plot(no_circ_conv(signal.gaussian(50, std=5), np.pad(np.ones(1), (24,25), mode='constant', constant_values=(0,0))));plt.savefig("output/no_circ_conv")


# sorry about the mess i was trying to do the whole thing in a line 
# and i guess i was succesful as the lambda definition is one line
# a spent a stupid amount of time trying to figure out how to call 
#define a function in one line but thats impossible it would appear
# anyways im just doubling the length of the array so it never wraps around
