import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import pywt
from scipy import signal

fliter = int(0.8*360)
Initial_intercept_point = 0
Final_intercept_point = 2000
Give_up_size = int(fliter/2)

# 读取本地的109号记录，从0到25000，通道0
record = wfdb.rdrecord('109/109', sampfrom=0, sampto=25000, physical=False, channels=[0, ])
print("record frequency：" + str(record.fs))
# 读取前2000数据
Initial_intercept_point = 0
Final_intercept_point = 2000
ventricular_signal = record.d_signal[Initial_intercept_point:Final_intercept_point].flatten()
ECG_baseline = medfilt(ventricular_signal,fliter+1)
Totality_Bias = np.sum(ECG_baseline[Give_up_size:-Give_up_size])/(Final_intercept_point - Initial_intercept_point-fliter)
Filtered_ECG = ventricular_signal - ECG_baseline
Final_Filtered_ECG = Filtered_ECG[Give_up_size:-Give_up_size]-Totality_Bias
print('signal shape: ' + str(ventricular_signal.shape))
# 绘制波形

plt.figure(figsize=(100, 8))
plt.subplot(4, 1, 1)
plt.ylabel("Original ECG signal")
plt.plot(ventricular_signal[Give_up_size:-Give_up_size])#输出原图像
plt.subplot(4, 1, 2)
plt.ylabel("ECG baseline")
plt.plot(ECG_baseline[Give_up_size:-Give_up_size])#输出基线轮廓
plt.subplot(4, 1, 3)
plt.ylabel("Filtered ECG signal")
plt.plot(Final_Filtered_ECG)#基线滤波结果
plt.subplot(4, 1, 4)
plt.ylabel("Filtered ECG signal baseline")
Filtered_ECG_signal_baseline = medfilt(Filtered_ECG, fliter+1)
plt.plot(Filtered_ECG_signal_baseline[Give_up_size:-Give_up_size])#基线滤波结果




# 读取第100条记录的annatation，前1000个点
signal_annotation = wfdb.rdann("109/109", "atr", sampfrom=0, sampto=3000)
# 打印标注信息
print("chan: " + str(signal_annotation.chan))
print("sample: " + str(signal_annotation.sample))
print("symbol: " + str(signal_annotation.symbol))
print("aux_note: " + str(signal_annotation.aux_note))


plt.show()


