"""A simple test logit model with 5,000 zones and random numbers for skims
Using Numba"""

import numba
from numba import jit, prange
import numpy as np
import timeit
import os

N_ZONES = 5000

c_ivtt = -0.025
c_ovtt = c_ivtt * 2.5
c_cost = -0.005

k_trn = -1.5
k_hwy = -1.0

def init_data():

    data = {}
    data["hwy_time"] = np.random.rand(N_ZONES, N_ZONES).astype(np.float32)
    data["trn_time"] = np.random.rand(N_ZONES, N_ZONES).astype(np.float32)
    data["trn_fare"] = np.random.rand(N_ZONES, N_ZONES).astype(np.float32)

    # a little warm up operation:
    tmp = data["hwy_time"] + data["trn_time"] * 0.5

    return data

def calc_hwy_util(data):

    # Compute hwy utility - very simple
    # hwy_util = data["hwy_time"] * c_ivtt + k_hwy

    hwy_util = numba_hwy(data["hwy_time"], c_ivtt, k_hwy)

    #print("hwy_util mean:", hwy_util.mean())

    return hwy_util

@jit(nopython=True, parallel=True)
def numba_hwy(hwy_time, c_ivtt, k_hwy):

    # Simple sequential
    # return k_hwy + hwy_time * c_ivtt

    # Allocate then add
    # rv = np.full(hwy_time.shape, k_hwy, dtype=np.float32)
    # rv += hwy_time * c_ivtt

    # With a loop
    rv = np.empty(hwy_time.shape, dtype=np.float32)
    for i in prange(hwy_time.shape[0]):
        for j in prange(hwy_time.shape[1]):
            rv[i, j] = k_hwy + hwy_time[i, j] * c_ivtt

    # Allocate then add in a loop
    # rv = np.full(hwy_time.shape, k_hwy, dtype=np.float32)
    # for i in prange(hwy_time.shape[0]):
    #     for j in prange(hwy_time.shape[1]):
    #         rv[i, j] += hwy_time[i, j] * c_ivtt

    return(rv)

if __name__ == '__main__':

    timer_number = 10

    #hwy_time = timeit.timeit(setup="data = init_data(); calc_hwy_util(data)", stmt="calc_hwy_util(data)", number=timer_number, globals=globals())
    hwy_time = timeit.timeit(setup="data = init_data(); a=data['hwy_time']; rv=np.empty(a.shape, dtype=np.float32); numba_hwy(a, c_ivtt, k_hwy)", stmt="numba_hwy(a, c_ivtt, k_hwy)", number=timer_number, globals=globals())

    print(f"Highway Time: {hwy_time}")

    # print(torch.__config__.parallel_info())