import numpy as np
import matplotlib.pyplot as plt

N = 128
n = np.arange(N)
k0 = 7
x = np.exp(1j*2*np.pi*k0*n/N) #complex sinusoid input

X = np.zeros(N) #Output

for k in range(N):
    s = np.exp(1j*2*np.pi*k*n/N) #complex sinusoid
    X[k] = sum(x*np.conjugate(s))

fig, ax = plt.subplots(2,1,sharex='col',sharey='row')
ax[0].plot(n,np.real(x),label='Real')
ax[0].plot(n,np.imag(x),label='Imaginary')
ax[0].set_title('Time Domain')
ax[0].grid()
ax[0].legend(fontsize = 4)

ax[1].plot(n,X)
ax[1].set_title('Frequency Domain')
ax[1].axis([0,N,0,N])
ax[1].grid()


plt.show()
    