import numpy as np
import matplotlib.pyplot as plt

def DFT(x):
     #real sinusoid input
    N = len(x)
    n = np.arange(-N/2,N/2)
    X = np.array([]) #Output
    for k in n:
        s = np.exp(1j*2*np.pi*k*n/N) #complex sinusoid
        X = np.append(X,sum(x*np.conjugate(s)))
    return(X)

N = 128
k0 = 20
n = np.arange(-N/2,N/2)
x = np.cos(2*np.pi*k0*n/N)
X = DFT(x)

#If we use 0 to N - 1, there will be two peaks of half the amplitude [(NA_0)/2] at k0 and N-k0

#If we use -N/2 to N/2 - 1, there will be peaks at k0 and -k0 of half the amplitude

fig, ax = plt.subplots(2,1,sharex='col',sharey='row')
ax[0].plot(n,np.real(x),label='Real')
ax[0].plot(n,np.imag(x),label='Imaginary')
ax[0].set_title('Time Domain')
ax[0].grid()
ax[0].legend(fontsize = 4)

ax[1].plot(n,X)
ax[1].set_title('Frequency Domain')
ax[1].axis([-N/2,N/2,0,N])
ax[1].grid()


plt.show()
    