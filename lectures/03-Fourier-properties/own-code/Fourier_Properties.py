import numpy as np
from scipy.signal import triang
from scipy.fftpack import fft

N = 15
x = triang(N) #Triangular signal with 15 samples

fftbuffer = np.zeros(N)
if N%2!=0:
    fftbuffer[:(N+1)/2] = x[N//2:] #First part will be second
    fftbuffer[(N+1)/2:] = x[:N//2] #Second part will be first
else:
    fftbuffer[:N//2] = x[N//2:]
    fftbuffer[N//2:] = x[:N//2]


X = fft(fftbuffer) #Automatically zero pads to nearest 2^n samples
mX = abs(X) 
pX = np.angle(X)