import numpy as np
from scipy.signal import triang
from scipy.fftpack import fft, fftshift

M = 14
N = int(2**np.ceil(np.log2(M)))
x = triang(M) #Triangular signal with 15 samples

fftbuffer = np.zeros(N)
if M%2!=0:
    fftbuffer[:M//2] = x[(M+1)//2:] #First part will be second
    fftbuffer[N-(M+1)//2:] = x[:(M+1)//2] #Second part will be first
else:
    fftbuffer[:M//2] = x[M//2:]
    fftbuffer[N-M//2:] = x[:M//2]


X = fft(fftbuffer) #Automatically zero pads to nearest 2^n samples
mX = abs(X) 
pX = np.angle(X)
print(x)
print(fftbuffer) #16 samples
print(fftshift(x)) #15 samples

x1 = np.append(x,np.zeros(N-M))
print(fftshift(x1))


print(len(mX))