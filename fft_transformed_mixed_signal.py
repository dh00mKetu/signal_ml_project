import numpy as np
import matplotlib.pyplot as plt
import mixed_signal


#the time domain values converted to frequency to frequency domain and we show the actual hidden signal in the noise as well
plt.plot(mixed_signal.positive_fft_signal_freq,mixed_signal.mag_fft_mixed_signal)
plt.xlabel("Frequency(Hz)",size=18)
plt.ylabel("Magnitude",size=18)
plt.title("Frequency Spectrum of Noisy Signal",size=24)
plt.grid(True,axis="both",linestyle="--")
plt.minorticks_on()


plt.show()
