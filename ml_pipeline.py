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

transformed_signal101=np.fft.fft(body_acc_x_train)
transformed_signal10=abs(transformed_signal101)
transformed_signal11=(transformed_signal10[:,0:(np.size(transformed_signal10,axis=1))//2])


fs=50
time=np.arange(0,2.56,1/fs)
freq1=np.fft.fftfreq(n=len(time),d=1/fs)
freq2=freq1[0:(len(freq1))//2]
#filter

b,a = signal.butter(3,8,'low',fs=fs)
filtered_body_acc_x_train=signal.filtfilt(b,a,body_acc_x_train)

#time features extraction
mean_body_acc_x_train=np.mean(body_acc_x_train,axis=1)
std_body_acc_x_train=np.std(body_acc_x_train,axis=1)
max_body_acc_x_train=np.amax(abs(body_acc_x_train),axis=1)


#frequency features
dom_freq_transformed_signal10=np.argmax(transformed_signal11,axis=1)
spectral_energy_transformed_signal11=np.sum(transformed_signal11**2,axis=1)
final= np.stack((mean_body_acc_x_train,std_body_acc_x_train,max_body_acc_x_train,dom_freq_transformed_signal10,spectral_energy_transformed_signal11),axis=1)


gss= GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
train_index,test_index=next((gss.split(final,y_text,subject_train)))
X_train=(final[train_index])
X_test=(final[test_index])
y_train=(y_text[train_index])
y_test=(y_text[test_index])

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

train_clf=clf.fit(X_train_scaled,y_train)
predict_clf=clf.predict(X_test_scaled)
print(classification_report(y_test,predict_clf))
print(confusion_matrix(y_test,predict_clf))