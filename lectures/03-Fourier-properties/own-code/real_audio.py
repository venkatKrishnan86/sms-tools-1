import numpy as np
from scipy.fftpack import fft
import os, sys, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models/'))
import utilFunctions as UF
import matplotlib.pyplot as plt

window_size = 501
if(window_size%2!=0):
    hM1 = int(math.floor((window_size+1)/2))
    hM2 = int(math.floor(window_size//2))
else:
    hM1 = int(math.floor(window_size//2))
    hM2 = int(math.floor(window_size//2 - 1))

(fs , x) = UF.wavread('../../../sounds/soprano-E4.wav')
x1 = x[5000:5000+window_size] * np.hamming(window_size)

t = np.arange(0,len(x)/fs,1/fs)

N = 2048 #FFT size - Larger the vaue, we get a SMOOTHER SPECTRUM as more interpolation points are present
fftbuffer = np.zeros(N)
fftbuffer[N-hM2:] = x1[:hM2]
fftbuffer[:hM1] = x1[hM2:]

F = fs/N
freq = np.arange(0,fs,F)

X = fft(fftbuffer)
mX = 20*np.log10(abs(X))
pX = np.unwrap(np.angle(X)) #Phase unwrapping

fig, ax = plt.subplots(3,1)

ax[0].plot(t,x)
ax[0].set_title('Soprano E4 note (330 Hz)',fontsize = 10)
ax[0].axis([0,len(x)/fs,-1,1])

ax[1].plot(freq[:N//2],mX[:N//2]) #Only half as its symmetric
ax[1].set_title('Magnitude Spectrum at t = %s s'%(round(t[5000],3)),fontsize = 10)
ax[1].axis([0,fs/2,min(mX),max(mX)])
#ax[1].set_xscale('log') #Convert lowest x value to a non zero in the ax[1].axis()

ax[2].plot(freq[:N//2],pX[:N//2]) #Only half as its anti-symmetric
ax[2].set_title('Phase Spectrum  at t = %s s'%(round(t[5000])),fontsize = 10)
ax[2].axis([0,fs/2,min(pX),max(pX)])
#ax[2].set_xscale('log') #Convert lowest x value to a non zero in the ax[2].axis()

plt.show()

