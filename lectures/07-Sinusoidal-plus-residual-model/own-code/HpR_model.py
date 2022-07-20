import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal import get_window
import matplotlib.pyplot as plt
import os, sys

from sklearn.datasets import fetch_20newsgroups_vectorized

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models'))
import utilFunctions as UF
import dftModel as DFT
import harmonicModel as HM

fs, x = UF.wavread('../../../sounds/oboe-A4.wav')
M = 801
N = 2048
t = -80
pin = 40000
minf0 = 300
maxf0 = 500
f0et  = 5
nH = 60
harmDevSlope = 0.001

w = get_window('blackman', M)

x1 = x[pin - (M+1)//2 : pin + M//2]
mX, pX = DFT.dftAnal(x1, w, N)
ploc = UF.peakDetection(mX, t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
ipfreq = iploc*fs/N
f0 = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0)
hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0, nH, [], fs, harmDevSlope)

Ns = 512
Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)

wr = get_window('blackmanharris', Ns)
xw2 = x[pin - Ns//2: pin + Ns//2] * wr/sum(wr)
X2 = fft(fftshift(xw2))
Xr = X2 - Yh

plt.plot(20*np.log10(abs(Yh[:100])), label = 'Synthesized')
plt.plot(20*np.log10(abs(X2[:100])), label = 'Original')
plt.plot(20*np.log10(abs(Xr[:100])), label = 'Residual')
plt.legend()

plt.show()




