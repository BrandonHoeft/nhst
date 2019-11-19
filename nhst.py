import numpy as np


def ecdf(data):
    """
    Calculate the empirical cumulative distribution function
    of a continuous variable. Takes an array-like object.
    Returns a tuple: an array of ascending sorted data, and 
    an array of corresponding cumulative % values. 
    """
    n = len(data)
    x = np.sort(data)
    # cumulative_p: proportion of data points that have a value less than 
    # corresponding x value (i.e. cumulative %)
    cumulative_p = np.arange(1, n+1) / n
    return x, cumulative_p


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute covariance matrix: variability due to codependence between 2 arrays
    cov_mat = np.cov(x, y)
    # account for variability inherent in each independent variable  
    r = cov_mat[0,1] / (np.sqrt(cov_mat[0,0]) * np.sqrt(cov_mat[1,1]))
    return r


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
    a function of interest. Useful for simulating a Confidence Interval.
    """
    #Initialize array of replicates: bs_replicates
    boot_replicates = np.empty(n)
    
    #Generate replicates
    for i in range(n):
        boot_replicates[i] = one_bootstrap_replicate(data=data, func=func)
        
    return boot_replicates
