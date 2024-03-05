import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy import signal

fliter = int(0.8*360)
Initial_intercept_point = 0
Final_intercept_point = 2000
Give_up_size = int(fliter/2)

record = wfdb.rdrecord('109', sampfrom=0, sampto=25000, physical=False, channels=[0, ])
print("record frequency：" + str(record.fs))

Original_ECG = record.d_signal[0:2000].flatten()

def high_fliter(Original_ECG, frequency=360, lowpass=1):
	#3是滤波器阶数
	#lowpass / frequency * 2 计算机截至频率
	#[b, a]为设计好的滤波器的系统函数的系数
    [b, a] = signal.butter(3, lowpass / frequency * 2, 'highpass')
	# 将设计好的系数和待滤波的信号扔进filtfilt中返回值为滤波之后的结果
    Signal_pro = signal.filtfilt(b, a, Original_ECG)
    return Signal_pro

fliter_ecg1 = high_fliter(Original_ECG)


def loss_fliter(Original_ECG, frequency=360, highpass=20):
    [b, a] = signal.butter(3, highpass / frequency * 2, 'lowpass')
    Signal_pro = signal.filtfilt(b, a, Original_ECG)
    return Signal_pro

fliter_ecg2 = loss_fliter(Original_ECG)
fliter_ecg3 = loss_fliter(fliter_ecg1)

plt.figure(figsize=(100, 8))
plt.subplot(4, 1, 1)
plt.ylabel("Original ECG signal")
plt.plot(Original_ECG)#输出原图像
plt.subplot(4, 1, 2)
plt.ylabel("Filtered ECG signal1")
plt.plot(fliter_ecg1)#去除基线漂移
plt.subplot(4, 1, 3)
plt.ylabel("Filtered ECG signal2")
plt.plot(fliter_ecg2)#去除工频干扰
plt.subplot(4, 1, 4)
plt.ylabel("Filtered ECG signal")
plt.plot(fliter_ecg3)#同时去除

# 读取第100条记录的annatation，前2000个点
signal_annotation = wfdb.rdann("109", "atr", sampfrom=0, sampto=3000)
# 打印标注信息
print("chan: " + str(signal_annotation.chan))
print("sample: " + str(signal_annotation.sample))
print("symbol: " + str(signal_annotation.symbol))
print("aux_note: " + str(signal_annotation.aux_note))

print(fliter_ecg3)
plt.show()
