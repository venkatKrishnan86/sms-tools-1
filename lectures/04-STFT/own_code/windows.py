import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt

M = 511 #[0,62]
window = get_window('hanning' , M)
hM1 = (M+1)//2 #32
hM2 = M//2 #31

N = 512
hN = N//2 #256
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = window[hM2:] #[0,31] = [31,62]
fftbuffer[N-hM2:] = window[:hM2] #[32,62] = [0,30]

X = fft(fftbuffer)
mX = 20*np.log10(abs(X))
pX = np.angle(X)

plt.figure(1)
plt.plot(fftbuffer)

mX1 = np.zeros(N)
mX1[:hN] = mX[hN:] #[0,255] (256) = [256,511] (256)
mX1[hN:] = mX[:hN] #[32,62] = [0,30]
pX1 = np.zeros(N)
pX1[:hN] = pX[hN:] #[0,255] (256) = [256,511] (256)
pX1[hN:] = pX[:hN]

plt.figure(2)
plt.plot((np.arange(-hN,hN)/N)*M,mX1 - max(mX1))
plt.axis([-31,31,-300,0])
plt.grid()
plt.show()

