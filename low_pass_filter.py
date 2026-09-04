import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#Day 1 progress: Generating a sine wave with additive Gaussian noise and plotting the time-domain signal.
#Day 2 Milestone: Applying the Fast Fourier Transform (FFT),
#plotting the frequency spectrum, and confirming the sharp signal peak at 5 Hz above the noise floor.

#creating the clean and the random noisy signal and mixing them to final mixed noise
np.set_printoptions(precision=2, floatmode='fixed')
fs=1000#sampling frequency
time=np.arange(0,1,1/fs) #this becomes our time axis

clean_wave=np.sin(2*np.pi*5*time)

np.random.seed(0)
pure_noise=np.random.normal(loc=0.0,scale=0.375,size=len(time))
final = np.add(clean_wave,pure_noise)#random mixed signal with noise and pure sound mixed


#applying the fourier transform to the random noise to change from time domain to frequency domain to separate our clean signal frequency
#from the noisy signal
mag10=np.fft.fft(final)
mag11= (mag10[0:(len(mag10))//2])
freq2=np.fft.fftfreq(n=(len(final)),d=1/fs)
mag=np.abs(mag11)
fourier_freq=(freq2[0:(len(freq2)//2)])




b,a=signal.butter(4,15,"low",fs=fs)
filtered_signal=signal.filtfilt(b,a,final)
plt.plot(time, clean_wave, label="Clean")
plt.plot(time, final, label="Noisy", alpha=0.5)
plt.plot(time, filtered_signal, label="Filtered")
plt.legend(fontsize=15)
plt.xlabel("Time (sec)",size = 18)
plt.ylabel("Amplitude",size = 18)
plt.title("Low-Pass Filtering: Noisy vs. Filtered Signal",size=24)
plt.grid(True)
plt.minorticks_on()
plt.show()