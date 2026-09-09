#Day 1 progress: Generating a sine wave with additive Gaussian noise and plotting the time-domain signal.
#Day 2 Milestone: Applying the Fast Fourier Transform (FFT),
#plotting the frequency spectrum, and confirming the sharp signal peak at 5 Hz above the noise floor.


import numpy as np
import matplotlib.pyplot as plt
#creating the clean and the random noisy signal and mixing them to final mixed noise

fs=1000#sampling frequency
time=np.arange(0,1,1/fs) #this becomes our time axis

clean_wave=np.sin(2*np.pi*5*time)

np.random.seed(0)
pure_noise=np.random.normal(loc=0.0,scale=0.375,size=1000)
final = np.add(clean_wave,pure_noise)#random mixed signal with noise and pure sound mixed

"""
plt.plot(time,clean_wave,label="Clean")
plt.plot(time,final,label="Noisy")

plt.legend(fontsize=15)
plt.xlabel("Time(sec)",size=18)
plt.ylabel("Amplitude",size=18)
plt.title("Clean Sine Wave vs. Noisy Signal(5 Hz + Gaussian Noise)",size=25)

"""
#applying the fourier transform to the random noise to change from time domain to frequency domain to separate our clean signal frequency
#from the noisy signal
mag10=np.fft.fft(final)
mag11= (mag10[0:(len(mag10))//2])
freq2=np.fft.fftfreq(n=(len(final)),d=1/fs)
mag=np.abs(mag11)
fourier_freq=(freq2[0:(len(freq2))//2])
#"""
plt.plot(fourier_freq,mag)
plt.xlabel("Frequency(Hz)",size=18)
plt.ylabel("Magnitude",size=18)
plt.title("Frequency Spectrum of Noisy Signal",size=24)
plt.grid(True,axis="both",linestyle="--")
plt.minorticks_on()
#"""

plt.show()
