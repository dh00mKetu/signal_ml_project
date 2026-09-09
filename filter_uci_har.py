import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


body_acc_x_train=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
y_text=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt")

transformed_signal101=np.fft.fft(body_acc_x_train)
transformed_signal10=abs(transformed_signal101)
transformed_signal11=(transformed_signal10[0:(len(transformed_signal10))//2])

fs=50
time=np.arange(0,2.56,1/fs)
freq1=np.fft.fftfreq(n=len(time),d=1/fs)
freq2=freq1[0:(len(freq1))//2]
#filter

b,a = signal.butter(3,8,'low',fs=fs)
filtered_body_acc_x_train=signal.filtfilt(b,a,body_acc_x_train)
plt.plot(time,filtered_body_acc_x_train[0],label="filtered")
plt.plot(time,body_acc_x_train[0],label="original")
plt.legend()
plt.show()
