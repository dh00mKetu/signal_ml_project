import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=2, floatmode='fixed')
#accelerometer data as the x axis
body_acc_x_train=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
#labels(the activities which are being performed, denoted by their no. not names, check uci har repo to match names with no.s)
y_text=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt")

subject_train=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/subject_train.txt")

fft_body_acc_x_train=np.fft.fft(body_acc_x_train)#converting the acc data from time domain to freq domain
mag_fft_body_acc_x_train=abs(fft_body_acc_x_train)
positive_mag_fft_body_acc_x_train=(mag_fft_body_acc_x_train[:,0:(np.size(mag_fft_body_acc_x_train,axis=1))//2])#cut the 128 sample window size to 64


fs=50 #decided on a sampling frequency, 50 intervals / sec, literally dividing a 1 sec time interval into 50 equal parts
time=np.arange(0,2.56,1/fs)#time array
positive_time=time[0:(len(time))//2]
fft_time_to_freq=np.fft.fftfreq(n=len(time),d=1/fs)#transformed it into frequency domain
positive_fft_time_to_freq=fft_time_to_freq[0:(len(fft_time_to_freq))//2]#removed the negative part



if __name__ == "__main__": 
    figure,axes = plt.subplots(1,2)
    axes[0].plot(time,body_acc_x_train[0],color="blue")
    axes[0].set_title("Time Spectrum of the Body Acc Data",fontsize=20)
    axes[0].set_xlabel("Time(sec)",size=18)
    axes[0].set_ylabel("Amplitude",size=18)
    axes[0].minorticks_on()
    axes[0].grid(visible=True,axis="both")


    axes[1].plot(positive_fft_time_to_freq,positive_mag_fft_body_acc_x_train[0],color="red")
    axes[1].set_title("Frequency Spectrum of the Body Acc Data",fontsize=20)
    axes[1].set_xlabel("Frequency",size=18)
    axes[1].set_ylabel("Magnitude",size=18)
    axes[1].minorticks_on()
    axes[1].grid(visible=True,axis="both")
    plt.show()


