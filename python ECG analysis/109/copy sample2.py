import matplotlib.pyplot as plt
import pywt
import wfdb
from scipy import signal

#Frequence 为信号的采样频率
#lowpass 高通滤波器可以通过的最低频率
record = wfdb.rdrecord('109/109', sampfrom=0, sampto=25000, physical=False, channels=[0, ])
print("record frequency：" + str(record.fs))

Original_ECG = record.d_signal[0:2000].flatten()

#预处理第一步：带通滤波
def high_fliter(Original_ECG, frequency=360, highpass=20,lowpass=1):
	#3是滤波器阶数
	#lowpass / frequency * 2 计算机截至频率
	#[b, a]为设计好的滤波器的系统函数的系数
    [b, a] = signal.butter(3, [lowpass / frequency * 2, highpass/frequency*2],'bandpass')
	# 将设计好的系数和待滤波的信号扔进filtfilt中返回值为滤波之后的结果
    Signal_pro = signal.filtfilt(b, a, Original_ECG)
    return Signal_pro


fliter_ecg = high_fliter(Original_ECG)#滤波

#画图
plt.plot(Original_ECG)
plt.plot(fliter_ecg)
plt.legend(['Before','After'])
print(fliter_ecg)
plt.show()
