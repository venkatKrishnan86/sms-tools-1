import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models'))

import utilFunctions as UF
import dftModel as DFT

fs, x  = UF.wavread('../../../sounds/oboe-A4.wav')
M=501
N=4096
t=-60

x1 = x[1000:1000+M]
window = get_window('hamming', M)
mX, pX = DFT.dftAnal(x1, window, N)
ploc = UF.peakDetection(mX,t)
iploc, ipmX, ippX = UF.peakInterp(mX, pX, ploc)

plt.plot(fs*np.arange(N/2+1)/N,mX)
plt.plot(fs*iploc/N, ipmX, 'x')
plt.xlim(0,fs/2)
plt.ylim(-100,0)
plt.show()
