import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=42,max_iter=1000)
from sklearn.metrics import classification_report,confusion_matrix

body_acc_x_train=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt")
y_text=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt")
subject_train=np.loadtxt("human+activity+recognition+using+smartphones/UCI HAR Dataset/train/subject_train.txt")


fs=50
time=np.arange(0,2.56,1/fs)
freq1=np.fft.fftfreq(n=len(time),d=1/fs)
freq2=freq1[0:(len(freq1))//2]
#filter

b,a = signal.butter(3,8,'low',fs=fs)
filtered_body_acc_x_train=signal.filtfilt(b,a,body_acc_x_train)

filtered_transformed_signal101=np.fft.fft(filtered_body_acc_x_train)
filtered_transformed_signal10=abs(filtered_transformed_signal101)
filtered_transformed_signal11=(filtered_transformed_signal10[:,0:(np.size(filtered_transformed_signal10,axis=1))//2])


#time features extraction
mean_filtered_body_acc_x_train=np.mean(filtered_body_acc_x_train,axis=1)
std_filtered_body_acc_x_train=np.std(filtered_body_acc_x_train,axis=1)
max_filtered_body_acc_x_train=np.amax(abs(filtered_body_acc_x_train),axis=1)


#frequency features
dom_freq_filtered_transformed_signal10=np.argmax(filtered_transformed_signal11,axis=1)
spectral_energy_filtered_transformed_signal11=np.sum(filtered_transformed_signal11**2,axis=1)

filtered_final=np.stack((mean_filtered_body_acc_x_train,std_filtered_body_acc_x_train,max_filtered_body_acc_x_train,dom_freq_filtered_transformed_signal10,spectral_energy_filtered_transformed_signal11),axis=1)
gss= GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
train_index,test_index=next((gss.split(filtered_final,y_text,subject_train)))


X_train=(filtered_final[train_index])
X_test=(filtered_final[test_index])
y_train=(y_text[train_index])
y_test=(y_text[test_index])

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

train_clf=clf.fit(X_train_scaled,y_train)
predict_clf=clf.predict(X_test_scaled)
print(classification_report(y_test,predict_clf))
print(confusion_matrix(y_test,predict_clf))

