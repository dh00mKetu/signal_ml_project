import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=42,max_iter=1000)
from sklearn.metrics import classification_report,confusion_matrix
import uci_human_act
import filter_uci_har

#Applying fft to the filtered data to be able to extract them for the frequency features
fft_filtered_body_acc_x_train=np.fft.fft(filter_uci_har.filtered_body_acc_x_train)
mag_fft_filtered_body_acc_x_train=abs(fft_filtered_body_acc_x_train)
positive_mag_fft_filtered_body_acc_x_train=(mag_fft_filtered_body_acc_x_train[:,0:(np.size(mag_fft_filtered_body_acc_x_train,axis=1))//2])


#time features extraction
mean_filtered_body_acc_x_train=np.mean(filter_uci_har.filtered_body_acc_x_train,axis=1)
std_filtered_body_acc_x_train=np.std(filter_uci_har.filtered_body_acc_x_train,axis=1)
max_filtered_body_acc_x_train=np.amax(abs(filter_uci_har.filtered_body_acc_x_train),axis=1)


#frequency features
dom_freq_mag_fft_filtered_body_acc_x_train=np.argmax(positive_mag_fft_filtered_body_acc_x_train,axis=1)
spectral_energy_positive_mag_fft_filtered_body_acc_x_train=np.sum(positive_mag_fft_filtered_body_acc_x_train**2,axis=1)
combined_features_filtered_body_acc_x_train=np.stack((mean_filtered_body_acc_x_train,std_filtered_body_acc_x_train,max_filtered_body_acc_x_train,dom_freq_mag_fft_filtered_body_acc_x_train,spectral_energy_positive_mag_fft_filtered_body_acc_x_train),axis=1)



gss= GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
train_index,test_index=next((gss.split(combined_features_filtered_body_acc_x_train,uci_human_act.y_text,uci_human_act.subject_train)))


X_train=(combined_features_filtered_body_acc_x_train[train_index])
X_test=(combined_features_filtered_body_acc_x_train[test_index])
y_train=(uci_human_act.y_text[train_index])
y_test=(uci_human_act.y_text[test_index])

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

train_clf=clf.fit(X_train_scaled,y_train)
predict_clf=clf.predict(X_test_scaled)
print(classification_report(y_test,predict_clf))
print(confusion_matrix(y_test,predict_clf))

