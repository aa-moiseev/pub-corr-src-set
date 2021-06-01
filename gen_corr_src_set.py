# --------------------------------------------------------------------------------------
# Generating signal waveforms with predefined correlations
#
# For detailed description of the method see:
#
#   A. Moiseev, "Simulating brain signals with predened mutual correlations"
#   doi:
#
# A.Moiseev, Behavioral and Cognitive Neuroscience Institute,
# Simon Fraser University, Canada
# May 2021
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Routines
# --------------------------------------------------------------------------------------
import math
import numpy as np
from numpy import NaN, ones, dot, reshape, concatenate, identity, hanning, cov
from numpy.linalg import inv, pinv, cholesky
#from scipy.signal import butter, lfilter
from scipy.signal import butter, filtfilt
from scipy.linalg import sqrtm

# --------------------------------------------------------------------------------------
# Check if an array is a symmetric positively defined matrix
# Returns True / False
# --------------------------------------------------------------------------------------
def is_pos_def(a):
# --------------------------------------------------------------------------------------
    try:
        cholesky(a)
        return True
    except:
        return False

# Defaults for the narrow-band noise signals
DEFAULT_FS = 100.
DEFAULT_BAND = (8.,12.)
DEFAULT_ORDER = 4
    
# --------------------------------------------------------------------------------------
# Generate n x m filtered white noise signal set
# Input:
#   n       number of signals
#   m       number of time points (m >> n)
#   fs      sampling rate (samples per second), default is defined by DEFAULT_FS constant
#   band    (fMin,fMax) frequency band boundaries, Hz (fMin < fMax), default is defined by DEFAULT_BAND
#   rand_seed
#           optional seed value to initialize the random number generator
#   order   Butterworth filter half order, default is set by DEFAULT_ORDER. Note: final filter order is
#           2*order
# Output:
#   s       (n x m) np-array of n filtered noise signals
# --------------------------------------------------------------------------------------
def gen_band_noise(n, m, fs = DEFAULT_FS, band = DEFAULT_BAND, rand_seed = None, order = DEFAULT_ORDER):
# --------------------------------------------------------------------------------------
    if fs <= 0:
        raise ValueError("Invalid sample rate specification: fs = {}".format(fs))

    if band[0]>=band[1]:
        raise ValueError("Invalid frequency band specification: {}".format(band))

    if order < 1:
        raise ValueError("Invalid filter order: {}".format(order))

    if rand_seed != None:
        np.random.seed(rand_seed)

    s = np.random.rand(n,m)
    s = s - dot(np.mean(s,axis=1,keepdims = True), np.ones((1,m)))    # Remove means

    # Filter to specified band
    nyq = 0.5 * fs
    band = np.array(band) / nyq                             # Convert to relative frequencies
    b, a = butter(order, [band[0], band[1]], btype='band')  # Design the filter
    s = filtfilt(b, a, s, axis = 1) 
    return s

# --------------------------------------------------------------------------------------
# Generate n x m filtered white noise signal set with a different frequency band for each signal
# Input:
#   n       number of signals
#   m       number of time points (m >> n)
#   fs      sampling rate (samples per second), default is defined by DEFAULT_FS constant
#   bands   A n-long list or a tuple of bands specifications. Each specification is a tuple: f_min_i, f_max_i).
#           For example, 2 bands may be specified as follows
#   rand_seed
#           optional seed value to initialize the random number generator, default - None
#   order   Butterworth filter half order, default is set by DEFAULT_ORDER. Note: final filter order is
#           2*order
# Output:
#   s       (n x m) np-array of n filtered noise signals
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def gen_multi_band_noise(n, m, fs = DEFAULT_FS, bands = None, rand_seed = None, order = DEFAULT_ORDER):
# --------------------------------------------------------------------------------------
    if bands == None:
        return gen_band_noise(n, m, fs, rand_seed = rand_seed, order = order)

    if len(bands) != n:
        raise ValueError('Length of bands list should be equal to n')

    if rand_seed != None:
        np.random.seed(rand_seed)   # Initialize random generator
        rand_seed = None            # Prevent gen_band_noise() to re-initialize it

    sig = np.zeros((n,m))

    for i in range(n):
        sig[i,:] = gen_band_noise(1, m, fs, bands[i], rand_seed, order).flatten()

    return sig

# --------------------------------------------------------------------------------------
# Generate (n x m) signal set with unit variances and a given correlation matrix
# All arrays are expected to be numpy arrays.
#
# Input:
#   n               number of signals
#   m               number of time points; m/n > MIN_RATIO
#   target_corr     (n x n) target correlation matrix, must be positively defined.
#                   If none is given, n-dimensional identity matrix will be used
#                   NOTE: if target corr is not a correlation matrix but is still a positively
#                   defined symmetric matrix, returned signals will not have unit variances
#   taper           taper function (m-dimensional vector), or None (default). In the latter case
#                   Hanning taper will be used 
#   evoked          (n x m) set of evoked parts of the target signals, or None (default). Evoked parts
#                   *cannot* be constant offsets independent of time - there should be some variation
#   quot            n-dimensional vector of quotients, or None. Each quotient should be a number in the
#                   [0, 1] range, which specifies relative contribution of the variance
#                   of the evoked part to the total variance of the generated signal. If not specified
#                   but evoked parts are provided, quotients = 0.5 will be used for all signals.
#   nepochs         Number of epochs to generate; if None (default) - a signle epoch will be generated
#   seed_sig        A function with a signature sig_seed(n, m, <args>, rand_seed = None) which returns
#                   a (n x m) numpy array.
#                   If not supplied, gen_band_noise() will be used
#   args, kwargs    A list of positional and key word arguments for sig_seed which will be passed after n,m.
#                   If none are given and seed_sig == gen_band_noise, then all defaults for gen_band_noise
#                   will be used
# Output:
#   nepochs x n x m numpy array of target signals
# --------------------------------------------------------------------------------------
def gen_corr_src_set(n, m,
        target_corr = None,
        taper = None,
        evoked = None,
        quot = None,
        nepochs = None,
        seed_sig = gen_band_noise,
        *args,
        **kwargs
        ):

# --------------------------------------------------------------------------------------
    MIN_RATIO = 3           # Minimal value of m/n to get meaningful covariance estimates
    DEFAULT_QUOTIENT = 0.5  # Default contribution of the evoked variance to the total variance

    # Sanity checks and preparations. m, n:
    if m/n < MIN_RATIO:
        raise ValueError("m / n should be larger than {}".format(MIN_RATIO))

    # target_corr:
    if np.all(target_corr == None):
        target_corr = identity(n)
    else:
        if target_corr.shape[0] != n:
            raise ValueError("target_corr shoud be a {} x {} array".format(n,n))
        
        if not is_pos_def(target_corr):
            raise ValueError("target_corr is not positively defined")
        
        if not np.allclose(np.diag(target_corr), np.ones((n,1))):
            print("\nWARNING: target_corr does not appear to be a correlation matrix.\n" + 
                    "Output signals may not be properly normalized.\n")

    # taper:            
    if np.all(taper == None):
        taper = hanning(m)
    else:
        taper = taper.flatten()

        if taper.shape[0] != m:
            raise ValueError("taper should have exactly {} elements".format(m))

    # evoked
    if np.all(evoked == None):
        no_evoked = True
    else:
        if (len(evoked.shape) != 2) or (evoked.shape[0] != n) or (evoked.shape[1] != m) :
            raise ValueError("evoked should be a ({} x {}) array".format(n,m))

        no_evoked = False

        # Normalize evoked timecourses
        cov_e = cov(evoked)
        var_e = np.diag(cov_e)

        for i in range(n):
            if np.allclose(var_e[i], np.zeros(n)):
                evoked[i] -= evoked[i,0]              # Remove constant offsets
            else:
                evoked[i] /= math.sqrt(var_e[i])    # Normalize evoked signals

        cov_e = cov(evoked)

    # quot
    if no_evoked:
        quot = np.zeros(n)      # It cannot be None, so set to 0s
    else:   # Evoked signals supplied
        if np.all(quot == None):
            quot = np.ones(n) * DEFAULT_QUOTIENT
        else:
            quot = quot.flatten()

            if quot.shape[0] != n:
                raise ValueError("quot should have exactly {} elements".format(n))

            if (not np.all(quot >= 0)) or (not np.all(quot <= 1)):
                raise ValueError("quot elements should all be in the [0,1] range")

    # nepochs
    if not nepochs:
        nepochs = 1
    else:
        nepochs = abs(nepochs)

    # Prepare Q, E matrices
    q_mtx = np.diag(np.sqrt(quot))
    e_mtx = np.diag(taper)
    e_sq = dot(e_mtx, e_mtx)

    # Verify that target corr is mathematically  possible and prepare m1 = (Ctarget - Q Cev Q)^1/2
    if no_evoked == False:
        m1 = target_corr - dot(q_mtx, dot(cov_e, q_mtx))

        if not is_pos_def(m1):
            raise ValueError("Target correlation matrix is not mathematically possible with current inputs.\n" +
                    "Try to decrease quot values or change evoked signals\n")
        m1 = sqrtm(m1)
    else:
        m1 = sqrtm(target_corr)

    # Construct the out-projector P
    if no_evoked:
        tmp = reshape(taper,(m,1))
    else:
        tmp = dot(e_mtx,evoked.T)   # There is no need to remove means from evoked
        tmp = concatenate((reshape(taper,(m,1)), tmp), axis = 1)

    p_mtx = identity(m) - dot(tmp, pinv(tmp))

    # Generate epochs
    res = np.full((nepochs,n,m), NaN)

    for i in range(nepochs):
        # If no info regarding seed signal function is supplied, use all the defaults for gen_band_noise
        if (seed_sig == gen_band_noise) and (len(args) == 0) and (len(kwargs) == 0):
            sp = dot(seed_sig(n,m,DEFAULT_FS,DEFAULT_BAND, rand_seed = None, order = DEFAULT_ORDER), p_mtx)
        else:
            sp = dot(seed_sig(n,m,*args,**kwargs), p_mtx)

        if i == 0:
            kwargs['rand_seed'] = None  # Prevent setting the seed for epochs # > 0, otherwise all epochs are the same

        spe = dot(sp,e_mtx)
        cov_spe = cov(spe)
        rt_spe = sqrtm(inv(cov_spe))

        if no_evoked:
            res[i] = dot(m1, dot(rt_spe, spe))
        else:
            res[i] = dot(m1, dot(rt_spe, spe)) + dot(q_mtx, evoked)

    return res

