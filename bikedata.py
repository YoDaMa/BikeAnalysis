import csv
import numpy as np
from scipy import signal
from matplotlib import *
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import sys
from pylab import *


"""
Digital Gym Research Group - Ashu Sabharwal
Written by Thomas (Tingkai) Liu and Yoseph Maguire

This code is designed to process raw csv files output from accelerometers
attached to a bike pedal, and output the overall cadence of the rider over
the dataset.

It needs to be updated to log cadence over time using SFFTs. 


"""

# Possibly use scipy.signal
"""
Just for speed testing purposes: be sure to include Bike_4XYZ.csv.
"""
if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename ='Bike_4XYZ.csv'


def bikedata(fname):

    X, Y, Z = retrieveacc("{}".format(fname)) 

    Y_Mean = float(np.mean(Y))
    Z_Mean = float(np.mean(Z))
    X_Mean = float(np.mean(X))

    Tot_Mean = Z_Mean+X_Mean+Y_Mean 
    Z_weight = Z_Mean/Tot_Mean
    X_weight = X_Mean/Tot_Mean
    Y_weight = Y_Mean/Tot_Mean

    Y_accel = []
    X_accel = []
    Z_accel = []

    for i in range(0,len(X)):
        X_accel.append(X[i]*X_weight)
        Y_accel.append(Y[i]*Y_weight)
        Z_accel.append(Z[i]*Z_weight)


    # data = np.sqrt(np.square(X_accel)+np.square(Y_accel)+np.square(Z_accel)) # Use if worried about orientation.
    data = Z

    D_mean = float(np.mean(data)) 
    for i in range(0,len(data)):
        data[i] -= D_mean
    Fs = 100 # 120 for new sensor

    

    np2 = nextpow2(len(data))
    fftlength = np2
    ctr = (fftlength/2) + 1
    faxis = np.multiply(Fs/2,np.linspace(0,1,ctr))
    fdata = np.fft.fft(data,fftlength) #/ len(data) # Possibly need to normalize by the length of data.
    mag = np.absolute(fdata)
    mag_max = 0
    mag_idx = 0
    for i in range(0,len(mag)):
        if mag[i] > mag_max:
            mag_max = mag[i]
            mag_idx = i/ctr*(Fs/2)
    rpm = mag_idx * 60

    winlen = len(data)//10
    np2w = nextpow2(winlen)

    f, t, Sxx = signal.spectrogram(data, fs=100, window='hamming', nperseg=winlen, noverlap=winlen//4, nfft=np2w,
                               detrend='constant', return_onesided=True, scaling='density')
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    print("Primary Sxx:",len(Sxx))
    print("Nested Sxx:", Sxx[0])
    print("Frequency:", len(f))
    print("Time:", len(t))
    peaks = []
    # peaks = [max(row[i]) for i in range(len(Sxx[0])) for row in Sxx]
    for i in range(len(Sxx[0])):
        cp = 0
        for row in Sxx:

            if cp < row[i]:
                cp = row[i]
        peaks.append(cp*60)

    tsxx = np.transpose(Sxx)
    print("Primary Sxx:", len(tsxx))
    print("Nested Sxx:", len(tsxx[0]))
    peaks = [(60*max(row)) for row in tsxx]
    print("Average RPM from Peaks:", mean(peaks))
    return rpm, peaks



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

def retrieveacc(fname="Bike_1.csv"):
    X = []
    Y = []
    Z = []
    # data = []
    with open(fname,newline= '') as data:
        # for line in f.readlines():
        #   fields = line.split('\t')
        #   if fields[0].isdigit():
        #       data.append(fields)
        reader = csv.DictReader(data)
        #print(reader)
        for row in reader:
            # if "X" in row: pass
            # else: 
            #   for x in row: x=int(x)
            try:
                row['X'] = float(row['X'])
                X.append((row['X']))
            except ValueError:
                pass
            try:
                row['Y'] = float(row['Y'])
                Y.append((row['Y']))
            except ValueError:
                pass
            try:
                row['Z'] = float(row['Z'])
                Z.append((row['Z']))
            except ValueError:
                pass



        
    return X, Y, Z



rpm, stfts = bikedata(filename)
print("Average RPM over Data:",str(rpm))
print("Peaks:",str(stfts))

