import numpy as np

def one_bootstrap_replicate(data, func):
    """
    Generate 1 bootstrap replicate computing your desired statistic
    by applying a function of your choice to 1 bootstrap resampling 
    of a 1D array of data. 
    """
    return func(np.random.choice(data, size=len(data)))


def draw_n_bootstrap_replicates(data, func, n=1):
    """
    Draw N bootstrap replicates, given a 1D array of data and
    a function of interest.
    """
    #Initialize array of replicates: bs_replicates
    boot_replicates = np.empty(n)
    
    #Generate replicates
    for i in range(n):
        boot_replicates[i] = one_bootstrap_replicate(data=data, func=func)
        
    return boot_replicates
