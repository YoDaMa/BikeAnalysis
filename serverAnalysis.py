import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



# times = pd.date_range('2016-07-19', periods=200, freq='.025sec')


# sample = {}


# filename = "SAMPLE.json"
# file = pd.read_json("{}".format(filename))
# file = pd.Series(d, name = '')

"""
Citation: "Analysis of Time and Frequency Domain Features
of Accelerometer Measurements" - Waltenegus Dargie, Tech U of Dresden

Time Domain Features: Mean, Zero Crossing Rate, Maxima/Minima, Autocorrelation,
Cross Correlation, Linear Correlation Coefficient, Standard Deviation.

Frequency Domain Features: Mean, Correlation, Spectral roll-off, Spectral centroid,
Spectral flux.


"""





def zeroCrossRate(vec):
    """
    Find the number of times the signal crosses the zero point
    Cases:
        Crosses from positive to negative
        Crosses from zero to value opposite that of the value before it.
    """

    count = 0
    for x in range(1,len(vec)):
        prod = vec[x] * vec[x-1]
        if vec[x] == 0 and vec[x-1] > 0 and \
                           vec[x+1] > 0:
            count += 1
        elif prod < 0:
            count += 1
    return count / (len(vec)-1)

def autocorrelation(vec, mean):
    """
    Calculating the autocorrelation using the Pearson product-movement correlation coefficient.
    Normalizes the autocorrelation.

    :param vec, mean:
    :return autocorr:
    """

    if type(vec) != np.ndarray: vec = np.asarray(vec)
    n = len(vec)

    autocorr = np.correlate(vec,vec, mode='same')[n // 2:]
    lengths = range(n, n // 2, -1)
    autocorr /= lengths
    autocorr /= autocorr[0]
    return autocorr

def frequencyDomain(vec):
    L = len(data)
    np2 = nextpow2(L)
    fftlength = np2 
    ctr = int((fftlength/2))
    faxis = np.multiply(Fs/2,np.linspace(0,1,ctr))
    b, a = signal.butter(4, [.01, .5], 'bandpass', analog=False)
    bp_vec = signal.lfilter(b, a, vec)
    fdata = np.fft.fft(bp_vec,fftlength)
    mag = abs(fdata[0:ctr])
    print(fdata[0:ctr])
    print(len(fdata)//2,ctr)
    return np.asarray(mag)

def nextpow2(n):
        """
        n = integer.
        Bike_1.csv = 38558
        Return: Next largest value that is equal to 2^x
        """

        n -= 1
        n |= n >> 1
        n |= n >> 2   
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16  
        n += 1  
        return n

def findPeaks(vec, Fs=None):
    """
    
    """
    if type(vec) != np.ndarray: vec = np.asarray(vec)
    
    max_idx = vec.argmax()
    if Fs == None: pass
    else: max_idx = max_idx / ctr * (Fs/2)

    return max_idx

def readData(dict1):
    ts = dict1.keys()
    np.sort(ts) # Hopefully this sorts the calendar dates from least to greatest.
    return ts


def compute(accdata):
    ts, itdata = readData(accdata)
    ux_x = np.array()
    ux_y = np.array()
    ux_z = np.array()
    ux = [ux_x, ux_y, ux_z]

    for time in ts:
        np.append(ux_x,itdata[time][0])
        np.append(ux_y,itdata[time][1])
        np.append(ux_z,itdata[time][2])

    pow = [np.linalg.norm(ux_x), np.linalg.norm(ux_y), np.linalg.norm(ux_z)]
    tsig = ux[pow.index(max(pow))]  # set data as the axis with the largest power.

    Fs = 40
    tmean = tsig.mean()
    zcrate = zeroCrossRate(tsig) #
    mcrate = zeroCrossRate(tsig-tmean)
    minima = tsig.min()
    maxima = tsig.max()
    variance = tsig.var()
    autocorr = autocorrelation(tsig,tmean)
    std = np.std(tsig)

    fsig = frequencyDomain(tsig)
    fmean = fsig.mean()
    fcorr = autocorrelation(fsig,fmean)
    fpeaks = signal.find_peaks_cwt(fsig)
    frelmax = signal.argrelmax(fsig)
    rpm = findPeaks(fsig, Fs)

    # Need to make sure these creation of arrays are valid since it's a mix of
    # numbers and arrays...
    # timevec = np.array([tmean,zcrate,mcrate,minima,maxima,variance,autocorr,std])
    # fvec = np.array([fsig,fmean,fcorr,fpeaks,frelmax,rpm])


    return float(rpm)

