#plotting the frequency spectrum, and confirming the sharp signal peak at 5 Hz above the noise floor.


import numpy as np
import matplotlib.pyplot as plt


#creating the clean and the random noisy signal and mixing them to final mixed noise

fs=1000 #sampling frequency
time=np.arange(0,1,1/fs) #this becomes our time axis

clean_wave=np.sin(2*np.pi*5*time)#pure clean sine wave of 5 Hz

np.random.seed(0)#to create to same kind of randmoness everywhere, kinda like minecraft worlds
pure_noise=np.random.normal(loc=0.0,scale=0.375,size=1000)
mixed_signal = np.add(clean_wave,pure_noise)#random mixed signal with pure noise and pure sound mixed

#applying the fourier transform to the random noise to change from time domain to frequency domain to separate our clean signal frequency
#from the noisy signal
fft_mixed_signal=np.fft.fft(mixed_signal)# this generates the y axis of the amplitude v/s frequency graph
positive_fft_mixed_signal= (fft_mixed_signal[0:(len(fft_mixed_signal))//2]) #this cuts the y axis values corresponding to -ve x axis 
fft_signal_freq=np.fft.fftfreq(n=(len(mixed_signal)),d=1/fs)# this is our x axis, the FREQUENCIIIEEES!!!!!
mag_fft_mixed_signal=np.abs(positive_fft_mixed_signal)#we need absolute values from the complex values of our amplitudes,since the signal is an complex no. (watch 3b1b)
positive_fft_signal_freq=(fft_signal_freq[0:(len(fft_signal_freq))//2])#again we removed the -ve x axis values

#graph of the mixed signal(we show the actual clean wave hidden in it as a reference for future)
if __name__ == "__main__": 
#we used the condition above, because we wanted to to import this file to other files, to use the variables defined here in them
#but at the same time prevent this entire code from executing, as that would result in multiple outputs, specifically the output of this file
#being printed right before the actual desired output of that code file


    plt.plot(time,clean_wave,label="Clean")
    plt.plot(time,mixed_signal,label="Noisy")
    plt.legend(fontsize=15)
    plt.xlabel("Time(sec)",size=18)
    plt.ylabel("Amplitude",size=18)
    plt.title("Clean Sine Wave v/s Noisy Signal(5 Hz + Gaussian Noise)",size=25)
    plt.grid(True,axis="both",linestyle="--")
    plt.minorticks_on()
    plt.show()