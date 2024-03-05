import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy import signal
import pywt

fliter = int(0.8*360)
Initial_intercept_point = 0
Final_intercept_point = 2000
Give_up_size = int(fliter/2)

record = wfdb.rdrecord('109/109', sampfrom=0, sampto=25000, physical=False, channels=[0, ])
print("record frequency：" + str(record.fs))

Original_ECG = record.d_signal[0:2000].flatten()
fliter = int(0.8*360)

#中值滤波法
def medfilt_ecg(original_ecg, window_size):
    # 确保窗大小变为奇数
    window_size = window_size+1 if window_size % 2 == 0 else window_size
    give_up_size = int(fliter / 2)
    ecg_baseline = medfilt(Original_ECG, window_size)
    totality_bias = np.sum(ecg_baseline[give_up_size:-give_up_size])/(len(original_ecg)-2*give_up_size)
    filtered_ecg = original_ecg - ecg_baseline
    final_filtered_ecg = filtered_ecg[give_up_size:-give_up_size]-totality_bias
    return final_filtered_ecg


plt.title("ventricular signal")
plt.ylabel('mv')
plt.plot(medfilt_ecg(Original_ECG, fliter))
plt.show()
