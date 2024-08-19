"""A simple test logit model with 5,000 zones and random numbers for skims"""

import torch
import numpy as np
import timeit
import os

torch.set_default_device('cpu')

# Set environment variables for OpenMP
# os.environ["OMP_NUM_THREADS"] = "8"  # Adjust this value based on your experiments
# os.environ["MKL_NUM_THREADS"] = "8"  # Adjust this value based on your experiments

torch.set_num_threads(1)

N_ZONES = 5000

c_ivtt = -0.025
c_ovtt = c_ivtt * 2.5
c_cost = -0.005

k_trn = -1.5
k_hwy = -1.0

def init_data():

    data = {}
    data["hwy_time"] = torch.rand(N_ZONES, N_ZONES)
    data["trn_time"] = torch.rand(N_ZONES, N_ZONES)
    data["trn_fare"] = torch.rand(N_ZONES, N_ZONES)

    # a little warm up operation:
    tmp = torch.add(data["hwy_time"], data["trn_time"], alpha=0.5)

    return data

def calc_hwy_util(data):

    # Compute hwy utility - very simple (0.25 sec for 10 runs) - but faster on CUDA
    # hwy_util = data["hwy_time"] * c_ivtt + k_hwy

    # Compute hwy utility, with the add function and alpha parameter (0.12 sec for 10 runs) - but slower on CUDA
    hwy_util = torch.add(k_hwy, data["hwy_time"], alpha=c_ivtt)

    #print("hwy_util mean:", hwy_util.mean())

    return hwy_util

def calc_trn_util(data):

    # Compute trn utility - very simple (0.5352 sec for 10 runs)
    # trn_util = data["trn_time"] * c_ivtt + data["trn_fare"] * c_cost + k_trn

    # Compute trn utility, with the add function and alpha parameter ()
    #Also minimizing the number of operations
    # -- This seems to be fastest both with CPU and GPU --
    trn_util = torch.add(k_trn, data["trn_time"], alpha=c_ivtt)
    trn_util.add_(data["trn_fare"], alpha=c_cost)

    # print("trn_util mean:", trn_util.mean())

    return trn_util

def calc_logsum(utils):

    # Calculate logsum - manual calcs
    # logsum = torch.log(torch.exp(utils[0]) + torch.exp(utils[1]))

    # Calculate logsum - logaddexp
    logsum = torch.logaddexp(utils[0], utils[1])

    # Calculate logusm - using torch.logsumexp
    # logsum = torch.logsumexp(utils, dim=0)   

    # print("logsum mean:", logsum.mean())
    
    return logsum

if __name__ == '__main__':

    timer_number = 10

    #hwy_time = timeit.timeit(setup="data = init_data()", stmt="calc_hwy_util(data)", number=timer_number, globals=globals())
    #trn_time = timeit.timeit(setup="data = init_data()", stmt="calc_trn_util(data)", number=timer_number, globals=globals())



    # with pre-stack to better test performance of different options
    logsum_time = timeit.timeit(setup="data = init_data(); utils = torch.stack((calc_hwy_util(data), calc_trn_util(data)))", stmt="calc_logsum(utils)", number=timer_number, globals=globals())

    #print(f"Highway Time: {hwy_time}")
    #print(f"Transit Time: {trn_time}")
    print(f"Logsum Time: {logsum_time}")

    # print(torch.__config__.parallel_info())