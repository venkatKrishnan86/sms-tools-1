import numpy as np
import sys, os
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models'))

import stft as STFT
import utilFunctions as UF
import matplotlib.pyplot as plt
from scipy.signal import get_window

input_file = '../../../sounds/flute-A4.wav'
window = 'blackman'
M = 801
N = 1024
H = 270

fs, x = UF.wavread(input_file)

w = get_window(window, M)

mX, pX = STFT.stftAnal(x, w, N, H)

plt.pcolormesh(mX.T)
plt.colorbar()
plt.show()



