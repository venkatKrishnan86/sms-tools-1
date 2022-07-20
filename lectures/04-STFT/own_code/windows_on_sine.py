import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import get_window
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models'))

from dftModel import dftAnal

fs = 44100.0
f = 5000.0
N = 1024
M = 101
x = np.cos(2*np.pi*f*np.arange(M)/fs)
w = get_window('blackmanharris', M)
mX, pX = dftAnal(x, w, N)

plt.plot(np.arange(0,fs/2,fs/N),mX - max(mX))
plt.show()

