import numpy as np
import scipy as sp
from numpy import fft
from matplotlib import pyplot as plt


plt.close()
plt.figure(0)
wind = np.hamming(12)
window = np.hamming(200)
A = fft.fft(window,2048) /25.5
mag = np.abs(fft.fftshift(A))
freq = np.linspace(-.5,.5,len(A))
response = 20 * np.log10(mag)
response = np.clip(response,-100,100)
plt.plot(freq,response)
plt.title("Hamming Window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")


B = plt.figure(1)
window = np.hamming(51)
plt.plot(window)
plt.title("Hamming Window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
B.canvas.set_window_title('Hamming Window')


plt.figure(2)
A = fft.fft(window, 2048) / 25.5
mag = np.abs(fft.fftshift(A))
freq=np.linspace(-.5,.5, len(A))
response = 20 * np.log10(mag)
response = np.clip(response, -100, 100)
plt.plot(freq, response)
plt.title("Frequency response of Hamming Window")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")
plt.axis('tight')
plt.show()

G = [4,3,3,2,1]

H = []
H.append(G)
H.append([G])

print(H)