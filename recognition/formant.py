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

def formant_show(y,sr,loc,spec):
    wlen = len(y)
    wlen2 = wlen // 2

    y_abs = np.log(np.abs(np.fft.fft(np.multiply(y, np.hamming(wlen))))[:wlen2])#画频谱用
    y_abs = np.abs(np.fft.fft(np.multiply(y, np.hamming(wlen))))[:wlen2]#画频谱用

    freq = [i*sr/wlen for i in range(wlen2)] # 生成频率轴
    fig = plt.figure()
    ax1=fig.add_subplot(211);ax1.plot(freq,y_abs,linewidth=0.1);ax1.set_title('freq');ax1.grid()#画频谱
    ax2=fig.add_subplot(212);ax2.plot(freq,spec);ax2.set_title('formant estimationg');ax2.grid()#画共振峰
    for i in range(len(loc)):
        #plt.subplot(2, 1, 2)
        ax2.plot([freq[loc[i]],freq[loc[i]]],[np.min(spec), spec[loc[i]]], '-.k') # 画标识线
        ax2.text(freq[loc[i]],spec[loc[i]],'({0:.2f},{1:.2f})'.format(freq[loc[i]],spec[loc[i]]))#写坐标
    #plt.show()



if __name__ == '__main__':
    path = "./input.wav"
    y, sr = librosa.load("./input.wav")

    y = lfilter([1, -0.99], [1], y) # 预加重
    cepstL = 6

    formant,loc,spec = get_formant(y,cepstL) # 返回共振峰值、共振峰位置、包络线

    formant_show(y,sr,loc,spec) # 画共振峰

    plt.show()


# def lpc_coeff(s, p):
#     """
#     :param s: 一帧数据
#     :param p: 线性预测的阶数
#     :return:
#     """
#     n = len(s)
#     # 计算自相关函数
#     Rp = np.zeros(p)
#     for i in range(p):
#         Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
#     Rp0 = np.matmul(s, s.T)
#     Ep = np.zeros((p, 1))
#     k = np.zeros((p, 1))
#     a = np.zeros((p, p))
#     # 处理i=0的情况
#     Ep0 = Rp0
#     k[0] = Rp[0] / Rp0
#     a[0, 0] = k[0]
#     Ep[0] = (1 - k[0] * k[0]) * Ep0
#     # i=1开始，递归计算
#     if p > 1:
#         for i in range(1, p):
#             k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
#             a[i, i] = k[i]
#             Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
#             for j in range(i - 1, -1, -1):
#                 a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
#     ar = np.zeros(p + 1)
#     ar[0] = 1
#     ar[1:] = -a[:, p - 1]
#     G = np.sqrt(Ep[p - 1])
#     return ar, G



# def Formant_Root(u, p, fs, n_frmnt):
#     """
#     LPC求根法的共振峰估计函数
#     :param u:
#     :param p:
#     :param fs:
#     :param n_frmnt:
#     :return:
#     """
#     ar, _ = lpc_coeff(u, p)
#     U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
#     const = fs / (2 * np.pi)
#     rts = np.roots(ar)
#     yf = []
#     Bw = []
#     for i in range(len(ar) - 1):
#         re = np.real(rts[i])
#         im = np.imag(rts[i])
#         fromn = const * np.arctan2(im, re)
#         bw = -2 * const * np.log(np.abs(rts[i]))
#         if fromn > 150 and bw < 700 and fromn < fs / 2:
#             yf.append(fromn)
#             Bw.append(bw)
#     return yf[:min(len(yf), n_frmnt)], Bw[:min(len(Bw), n_frmnt)], U
