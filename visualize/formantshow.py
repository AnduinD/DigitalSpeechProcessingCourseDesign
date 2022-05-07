import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import lfilter

def get_formant(u, cepstL):
    """
    倒谱法共振峰估计函数
    param u:输入信号
    param cepstL:频率上窗函数的宽度
    return: val共振峰幅值 
    return: loc共振峰位置 
    return: spec包络线
    """
    wlen2 = len(u) // 2  # 频谱长度（采样点数的一半）
    U = np.log(np.abs(np.fft.fft(u)[:wlen2]))  # 取傅里叶对数谱
    Cepst = np.fft.ifft(U)   # 取倒谱
    cepst = np.zeros(wlen2, dtype=np.complex)  # 分配内存
    cepst[:cepstL] = Cepst[:cepstL]   # 倒谱加窗
    cepst[-cepstL + 1:] = Cepst[-cepstL + 1:] # 倒谱加窗
    spec = np.real(np.fft.fft(cepst[:wlen2])) # 取包络线  #然而这个逆变换并取不到包络线唉
    val, loc = local_maxium(spec)  #找极大值
    return val, loc, spec



def local_maxium(x):
    """
    求序列的极大值
    param x: 序列
    return maxium: 极大值序列
    return loc: 极大值点的位置序列
    """
    d = np.diff(x) # 一阶差分（相当于求导
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d - 1):#线性搜索
        if d[i] > 0 and d[i + 1] <= 0: #查找差分前后单调性改变的点
            maxium.append(x[i + 1])
            loc.append(i + 1)
    return maxium, loc


if __name__ == '__main__':
    path = "./input.wav"
    y, sr = librosa.load("./input.wav")
    
    y = lfilter([1, -0.99], [1], y) # 预加重
    cepstL = 6
    wlen = len(y)
    wlen2 = wlen // 2

    formant,loc,spec = get_formant(y,cepstL) # 返回共振峰值、共振峰位置、包络线

    #y_abs = np.log(np.abs(np.fft.fft(np.multiply(y, np.hamming(wlen))))[:wlen2])#画频谱用
    y_abs = np.abs(np.fft.fft(np.multiply(y, np.hamming(wlen))))[:wlen2]#画频谱用
    freq = [i*sr/wlen for i in range(wlen2)] # 生成频率轴
    plt.subplot(2, 1, 1);plt.plot(freq, y_abs,linewidth=0.1);plt.title('freq');plt.grid() # 画频谱
    plt.subplot(2, 1, 2);plt.plot(freq, spec);plt.title('formant estimationg');plt.grid() # 画共振峰
    for i in range(len(loc)):
        plt.subplot(2, 1, 2)
        plt.plot([freq[loc[i]],freq[loc[i]]],[np.min(spec), spec[loc[i]]], '-.k') # 画标识线
        plt.text(freq[loc[i]],spec[loc[i]],'({0:.2f},{1:.2f})'.format(freq[loc[i]],spec[loc[i]]))#写坐标
    plt.show()