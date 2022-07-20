import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.signal import get_window

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../software/models'))

import utilFunctions as UF
import harmonicModel as HM

fs, x = UF.wavread('../../../sounds/oboe-A4.wav')
time = len(x)/fs
print(time)

M = 2001
N = 4096
window = get_window('blackman', M)

t = -60 #threshold
minf0 = 300
maxf0 = 500 
f0et = 1 # max error allowed in f0 deviation
H = 1000 # hop size

f0 = HM.f0Detection(x, fs, window, N, H, t, minf0, maxf0, f0et)
print(len(f0))

plt.plot(np.arange(len(f0))*H/fs,f0)
plt.xlabel('Time (s)')
plt.ylabel('F0')
plt.axis([0,time, min(f0[2:-2]), max(f0)])
plt.show()