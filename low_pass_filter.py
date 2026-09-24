import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import mixed_signal

np.set_printoptions(precision=2, floatmode='fixed')

#applying the fourier transform to the random noise to change from time domain to frequency domain to separate our clean signal frequency
#from the noisy signal

b,a=signal.butter(4,15,"low",fs=mixed_signal.fs)
filtered_signal=signal.filtfilt(b,a,mixed_signal.mixed_signal)


plt.plot(mixed_signal.time, mixed_signal.clean_wave, label="Clean")
plt.plot(mixed_signal.time, mixed_signal.mixed_signal, label="Noisy", alpha=0.5)
plt.plot(mixed_signal.time, filtered_signal, label="Filtered")
plt.legend(fontsize=15)
plt.xlabel("Time (sec)",size = 18)
plt.ylabel("Amplitude",size = 18)
plt.title("Low-Pass Filtering: Noisy vs. Filtered Signal",size=24)
plt.grid(True)
plt.minorticks_on()
plt.show()