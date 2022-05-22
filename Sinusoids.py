from ast import increment_lineno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#real sinusoid
A = 0.8 #Amplitude
f0 = 1000 #Frequency of sinusoid
phi = np.pi/2 #Phase
fs = 44100 #Sampling Rate
t = np.arange(-0.002,0.002,1.0/fs) #time array

x = A*np.cos(2*np.pi*f0*t+phi) #Input signal x[n]

plt.plot(t,x)
plt.axis([min(t),max(t),-A,A])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Real Sinusoid')
plt.grid()
#complex sinusoid
N = 500
k = 1
n = np.arange(-N/2,N/2)
z = np.exp(1j*2*np.pi*k*n/N)

plt.figure(2)
plt.plot(n,np.real(z))
plt.plot(n,np.imag(z))
plt.axis([-N/2,N/2,-1,1])
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Complex Sinusoid')
plt.grid()

plt.show()

