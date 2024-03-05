import matplotlib.pyplot as plt
import wfdb
from scipy import signal
import numpy as np
from numpy import *
import pandas as pd



#打开文件读取数据
datafile = "ECGLog-2023-10-21-20-48-34.csv"
data = pd.read_csv(datafile)

#设置变量
start = 1000#开始点
sample_rate = 510#采样率
sample_count = 3000#切片长度
#切片
Original_ECG = data.iloc[start:start+sample_count,1]


t = np.linspace(0,sample_count/sample_rate,sample_count)

#ECG_line = plt.plot(t, Original_ECG, label='ECG')

#预处理第一步：带通滤波
def high_fliter(Original_ECG, frequency=360, highpass=25,lowpass=15):
	#3是滤波器阶数
	#lowpass / frequency * 2 计算机截至频率
	#[b, a]为设计好的滤波器的系统函数的系数
    [b, a] = signal.butter(3, [lowpass / frequency * 2, highpass/frequency*2],'bandpass')
	# 将设计好的系数和待滤波的信号扔进filtfilt中返回值为滤波之后的结果
    Signal_pro = signal.filtfilt(b, a, Original_ECG)
    return Signal_pro


fliter_ecg = high_fliter(Original_ECG)#滤波,此时查询到的类型是numpy.ndarray

# 双斜率处理
frequency=360
N=3000

a=int(0.015*frequency) # 两侧目标区间0.015~0.060s;
b=int(0.060*frequency)

Ns=N-2*b           # 确保在不超过信号长度；
S_l=zeros((b-a+1))
S_r=np.zeros((b-a+1))
s_d=np.zeros((Ns))

for i in range(1,Ns):        # 对每个点双斜率处理；
    for k in range(a,b):

        c=fliter_ecg[i+b]
        d=fliter_ecg[i+b-k]
        f=fliter_ecg[i+b+k]
        S_l[k-a+1] = (c-d)/k
    S_r[k-a+1] = (c-f)/ k
    S_lmax = S_l.max()
    S_lmin = S_l.min()
    S_rmax = S_r.max()
    S_rmin = S_r.min()
    C1 = S_rmax - S_lmin
    C2 = S_lmax - S_rmin
    s_d[i]= max(C1,C2)
print()

#低通滤波
def loss_fliter(data, frequency=360, highpass=15):
    [b, a] = signal.butter(3, highpass / frequency * 2, 'lowpass')
    Signal_pro2 = signal.filtfilt(b, a, s_d)
    return Signal_pro2
fliter_ecg2 = loss_fliter(s_d)

#滑动窗口积分
w=8
wd=7
e=zeros(w)
dl1 = np.append(e,fliter_ecg2)
dl2 = np.append(dl1,e)
dl3=dl2.astype(int)
m = zeros((Ns))
for n in range(w+1,Ns+w):
    m[n-w]= dl3[n-w:n+w].sum()
print(m)

f=np.ones(wd)
g=f*m[1]
h=f*27 #复现代码发现m[Ns]老是报错，所以干脆把整个列表打印查看最后一个元素是9

m_l1=np.append(g,m)
m_l2=np.append(m_l1,h)


#plt.plot(Original_ECG)
plt.plot(m_l2)
plt.show()