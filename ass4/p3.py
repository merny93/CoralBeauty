from p1 import my_shift
from p2 import my_correlation, my_gauss_window
import numpy as my_np

def my_shifted_correlation(my_array, my_shift_value):
    my_shifted_array = my_shift(my_array, my_shift_value)
    my_result = my_correlation(my_array, my_shifted_array)
    return my_result


my_gauss = my_gauss_window(50, 5)

my_gauss_shifted_correlation = my_shifted_correlation(my_gauss, 15)

import matplotlib.pyplot as plt
plt.clf()
plt.plot(my_gauss_shifted_correlation)
plt.savefig("output/my_gauss_shifted_correlation")

#almost the same explanation as for the last problem buttttt now we shifted one of them to begin with. This means that 
# they are the same when u shift it back so the correlation is not strongest at zero its strongest at the inverse of the shift.
# hence a shifted gaussian is what we would expect. 
