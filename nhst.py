import numpy as np

def one_bootstrap_replicate(data, func):
    """
    Generate 1 bootstrap replicate by applying a function of your choice
    to one bootstrap resampling of a a one-dimensional array. 
    """
    return func(np.random.choice(data, size=len(data)))



