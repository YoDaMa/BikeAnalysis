import csv
import numpy as np
from scipy import signal
from matplotlib import *
import matplotlib.pyplot as plt
import serverAnalysis
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import os

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




def bikedata(fname):

    sMat = retrieveacc("{}".format(fname))
    # User Acceleration
    ux_x = sMat['user_acc_x']
    ux_y = sMat['user_acc_y']
    ux_z = sMat['user_acc_z']
    # ux_x = sMat['user_acc_x(G)']
    # ux_y = sMat['user_acc_y(G)']
    # ux_z = sMat['user_acc_z(G)']
    ux = [ux_x,ux_y,ux_z]

    X_Mean = float(np.mean(ux_x))
    Y_Mean = float(np.mean(ux_y))
    Z_Mean = float(np.mean(ux_z))
    pow = [np.linalg.norm(ux_x),np.linalg.norm(ux_y),np.linalg.norm(ux_z)]
    # data = ux[pow.index(max(pow))] # set data as the axis with the largest power.
    data = ux_x
    # data = np.sqrt(np.square(X_accel)+np.square(Y_accel)+np.square(Z_accel)) # Use if worried about orientation.
    sxx = stft(data)
    Fs = 100
    var = np.mean((data-np.mean(data))**2) 

    plotstft(sxx)

    
    dlen = len(data)
    np2 = nextpow2(dlen)
    fftlength = np2
    ctr = int((fftlength/2))
    print(ctr)
    faxis = np.multiply(Fs/2,np.linspace(0,1,ctr))
    b, a = signal.butter(4, [.01, .5], 'bandpass', analog=False)
    filt_d = signal.lfilter(b, a, data)
    fild_d = data
    fdata = np.fft.fft(filt_d,fftlength) #/ len(data) # Possibly need to normalize by the length of data.
    mag = abs(fdata[0:ctr])
    print(fdata[0:ctr])
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(faxis,mag, 'b-',linewidth=2,label=r'$y=\sin(x)$')
    # ax.set_ylabel(r'$y$',fontsize=40)
    # ax.set_xlabel(r'$x$',fontsize=40)
    # ax.legend(loc='best',fontsize=40)
    # ax.grid(True)
    # fig.suptitle(r'$The\ Frequency\ Domain$',fontsize=40)
    # # fig.tight_layout(pad=0)
    plt.show()
    # fig.savefig('filename.png',dpi=125)



    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 40}


    rc('font', **font)
    # plt.plot(faxis, mag)
    # plt.title("Fourier Transform")
    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")
    # plt.show()
    mag_max = 0
    mag_idx = 0
    for i in range(0,len(mag)):
        if mag[i] > mag_max:
            mag_max = mag[i]
            mag_idx = i/ctr*(Fs/2)
    rpm = mag_idx * 60

    # winlen = len(data)//10
    # np2w = nextpow2(winlen)

    # f, t, Sxx = signal.spectrogram(data, fs=100, window='hamming', nperseg=winlen, noverlap=winlen//4, nfft=np2w,
    #                            detrend='constant', return_onesided=True, scaling='density')
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # print("Primary Sxx:",len(Sxx))
    # print("Nested Sxx:", Sxx[0])
    # print("Frequency:", len(f))
    # print("Time:", len(t))
    # peaks = []
    # # peaks = [max(row[i]) for i in range(len(Sxx[0])) for row in Sxx]
    # for i in range(len(Sxx[0])):
    #     cp = 0
    #     for row in Sxx:

    #         if cp < row[i]:
    #             cp = row[i]
    #     peaks.append(cp*60)

    # tsxx = np.transpose(Sxx)
    # print("Primary Sxx:", len(tsxx))
    # print("Nested Sxx:", len(tsxx[0]))
    # peaks = [(60*max(row)) for row in tsxx]
    # print("Average RPM from Peaks:", mean(peaks))
    # return rpm, peaks

    return rpm, 1

def stft(data,Fs=100):
    """
    Input: 3 Axis Data
    Output: Power Representation of
    """
    # Remove the DC Component of the function
    data = noDC(data)
    wlen = Fs * 10 # Window length = 10
    segs = len(data) // wlen # integer divsion to return number of segments of data
    windsegs = []
    numz = nextpow2(wlen) - wlen
    j = 0
    win = []
    for i in data:
        if j < wlen:  # append the value from j=0 to j = (number of samples - 1)
            win.append(i)
        else:
            j = 0
            for i in range(0, numz):  # Zero padding the window
                win.append(0)
            windsegs.append(win)  # Add that window to the segmented dataset
            win = [i]  # Reset the window variable
        j += 1
    b, a = signal.butter(4, [.01, .5], 'bandpass')
    dft=[]
    winlen = int(len(windsegs[0]))
    for seg in windsegs:
        seg = signal.lfilter(b, a, seg)
        wind = signal.get_window(('kaiser', 4.0), winlen)  # Beta Parameter of Kaiser: 4. Num Samples in window: 9.
        snip = seg * wind
        nfft = nextpow2(wlen)
        A = np.fft.fft(snip, nfft)
        dft.append(A)
    return dft

def noDC(data):
    D_mean = float(np.mean(data))
    for i in range(0, len(data)):
        data[i] -= D_mean
    return data

def plotstft(sxx, Fs=100):
    winlen = int(len(sxx[0]))

    # with plt.():
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    ctr = int(winlen / 2)
    faxis = np.multiply(Fs / 2, np.linspace(0, 1, ctr))*60
    ratio = []
    for dft in sxx:
        mag = abs(dft[0:ctr])
        max_idx, max_val = max(enumerate(mag), key = lambda p: p[1])
        ptotal = np.square(np.linalg.norm(mag,2))
        pmax = np.square(np.absolute(max_val))
        # print('max power: {}'.format(max_val))
        frac = pmax/ptotal
        # print(frac)
        ratio.append(frac)

        ax.plot(faxis, mag, linewidth=3.0)
        ax.set_xlabel('Frequency (RPM)')
        ax.set_ylabel('|H(f)|')
        ax.set_title('User 4 STFT Spectrum')
        font = {'family' : 'sans-serif ',
                'weight' : 'bold',
                'size'   : 30}
        rc('font', **font)

    fig1.savefig('STFTPlot.png')
    fig2 = plt.figure()
    plt.stem(np.linspace(1,len(ratio),num=len(ratio)),ratio, linewidth=3.0)
    plt.xlabel('Window (10s)', fontsize = 60)
    plt.ylabel('Symmetry in Pedaling', fontsize = 60)
    fig2.savefig('RatioPlot.png')
        # print("Ratio:{}".format(ratio))



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

def retrieveacc(fname):
    sMat = {} # Storage Matrix is a dictionary I suppose
    fpath = os.path.join(os.path.dirname(__file__),fname)
    with open(fpath,mode='r',newline=None) as csvfile:

        parsed = csv.reader(csvfile)
        parsed = np.array(list(parsed))
        titles, parsed = parsed[0], parsed[1:]
        data = parsed.transpose()
        data = cleandata(data[0:16]) # 16 columns because the other ones aren't useful.
        cidx = 0
        for col in data:
            sMat[titles[cidx]] = col
            cidx += 1
    return sMat


def quicksort(tarr):
    """
    Input must be a enumerated list with the index in [0] and value in [1]
    """
    if len(tarr) <= 1:
        return tarr
    else:
        l, pivot, r = partition(tarr)
        return quicksort(l)+[pivot]+quicksort(r)

def partition(tarr):
    pivot, tarr = tarr[len(tarr)-1], tarr[:len(tarr)-1]
    l = [x for x in tarr if x[1] <= pivot[1]]
    r = [x for x in tarr if x[1] > pivot[1]]
    return l, pivot, r

def cleandata(data):
    """
    Data needs to be represented as columns.
    Assumes uniform breaks, so that removing all empty values
    will uniformly shrink the data.
    """
    parsed = np.array([[float(x) for x in col if x != ''] for col in data])
    return parsed

def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Did not enter a filename.")
    else:
        rpm, stfts = bikedata(filename)
        print("Average RPM over Data:", str(rpm))
        print("Peaks:", str(stfts))



main()
