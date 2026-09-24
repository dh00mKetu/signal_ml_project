import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import uci_human_act


#filter
b,a = signal.butter(3,8,'low',fs=uci_human_act.fs)
filtered_body_acc_x_train=signal.filtfilt(b,a,uci_human_act.body_acc_x_train)



if __name__ == "__main__": 
    plt.plot(uci_human_act.time,filtered_body_acc_x_train[0],label="Filtered",color="red")
    plt.plot(uci_human_act.time,uci_human_act.body_acc_x_train[0],label="Original",color="blue")
    plt.xlabel("Time(sec)",size=18)
    plt.ylabel("Amplitude",size=18)
    plt.title("Body Acc: Original vs. Filtered Signal — Window 0(First)",size=24)
    plt.legend()
    plt.grid(True,axis="both",linestyle="--")
    plt.minorticks_on()
    plt.show()