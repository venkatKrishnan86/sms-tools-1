import numpy as np
import matplotlib.pyplot as plt

fs = 128
t = np.arange(-1,1,1.0/fs)
x = 0.2*np.sin(2*np.pi*60*t+9) + 0.4*np.sin(2*np.pi*40*t+8)

N = len(x)
n = np.arange(-N/2,N/2)
X = np.array([]) #Output
for k in n:
    s = np.exp(1j*2*np.pi*k*n/N) #complex sinusoid
    X = np.append(X,sum(x*np.conjugate(s)))

X = np.abs(X)/fs
f = np.arange(0,fs/2,fs/len(t))

print(len(t)==len(f))

fig, ax = plt.subplots(2,1)
ax[0].plot(t[fs:],x[fs:])
ax[1].plot(f,X[fs:])

plt.show()