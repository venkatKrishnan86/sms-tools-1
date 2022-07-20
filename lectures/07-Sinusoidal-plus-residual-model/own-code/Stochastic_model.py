from cmath import exp
import numpy as np
from scipy.signal import get_window, resample
import os, sys
from scipy.fftpack import fft, ifft
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models'))
import utilFunctions as UF
import matplotlib.pyplot as plt

fs, x = UF.wavread('../../../sounds/ocean.wav')
M = N = 256
stocf = 0.2 #Stochastic Factor - Downsampled by this value
w = get_window('hanning',M)
xw = x[10000:10000+M]*w
X = fft(xw)
mX = 20*np.log10(abs(X[:N//2]))
mXenv = resample(np.maximum(-200,mX),int(mX.size * stocf))

mY = resample(mXenv, N//2)
pY = 2*np.pi*np.random.rand(N//2)

plt.plot(mX,alpha = 0.4, label = 'Actual')
plt.plot(np.arange(0,mXenv.size/stocf,1/stocf),mXenv, alpha = 0.7, label = 'Downsampled')
plt.plot(mY, label = 'Smoothened Downsampled')
plt.legend()

Y = np.zeros(N, dtype=complex)
Y[:N//2] = 10**(mY/20)*np.exp(1j*pY)
Y[N//2:] = 10**(mY[::-1]/20)*np.exp(-1j*pY[::-1])
y = np.real(ifft(Y))

plt.figure(2)
plt.plot(xw,label = 'Actual')
plt.plot(y, label = 'Modelled')
plt.legend()

plt.show()
