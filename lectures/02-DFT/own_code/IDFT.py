import numpy as np
import matplotlib.pyplot as plt

N = 128
n = np.arange(-N/2,N/2)
k = n
k0 = 7.6
x = np.cos(2*np.pi*k0*n/N) #real sinusoid input

X = np.array([]) #Output
x_new = np.array([]) #newly formed input

for i in k:
    s = np.exp(1j*2*np.pi*i*n/N) #complex sinusoid
    X = np.append(X,sum(x*np.conjugate(s)))

for i in n:
    s = np.exp(1j*2*np.pi*k*i/N) #complex sinusoid
    x_new = np.append(x_new,sum(X*s)/float(N))


#If we use 0 to N - 1, there will be two peaks of half the amplitude [(NA_0)/2] at k0 and N-k0

#If we use -N/2 to N/2 - 1, there will be peaks at k0 and -k0 of half the amplitude



fig, ax = plt.subplots(3,1,sharex='col',sharey='row')
ax[0].plot(n,np.real(x),label='Real')
ax[0].plot(n,np.imag(x),label='Imaginary')
ax[0].set_title('Time Domain')
ax[0].grid()
ax[0].axis([-N/2,N/2,min(x),max(x) ])
ax[0].legend(fontsize = 4)

ax[1].plot(n,X)
ax[1].set_title('Frequency Domain')
ax[1].axis([-N/2,N/2,0,N])
ax[1].grid()

ax[2].plot(n,x_new)
ax[2].set_title('Recreated Time Domain')
ax[2].axis([-N/2,N/2,min(x_new),max(x_new)])
ax[2].grid()


plt.show()
    