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


def lpc_coeff(s, p):
    """
    :param s: 一帧数据
    :param p: 线性预测的阶数
    :return:
    """
    n = len(s)
    # 计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))
    # 处理i=0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0
    # i=1开始，递归计算
    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    ar = np.zeros(p + 1)
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    G = np.sqrt(Ep[p - 1])
    return ar, G

def Formant_Root(u, p, fs, n_frmnt):
    """
    LPC求根法的共振峰估计函数
    :param u:
    :param p:
    :param fs:
    :param n_frmnt:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    const = fs / (2 * np.pi)
    rts = np.roots(ar)
    yf = []
    Bw = []
    for i in range(len(ar) - 1):
        re = np.real(rts[i])
        im = np.imag(rts[i])
        fromn = const * np.arctan2(im, re)
        bw = -2 * const * np.log(np.abs(rts[i]))
        if fromn > 150 and bw < 700 and fromn < fs / 2:
            yf.append(fromn)
            Bw.append(bw)
    return yf[:min(len(yf), n_frmnt)], Bw[:min(len(Bw), n_frmnt)], U


if __name__ == '__main__':
    path = "./input.wav"
    y, sr = librosa.load("./input.wav")
    #y = y[:50000]
    
    y = lfilter([1, -0.99], [1], y) # 预加重
    cepstL = 6
    wlen = len(y)
    wlen2 = wlen // 2

    y_abs = np.log(np.abs(np.fft.fft(np.multiply(y, np.hamming(wlen))))[:wlen2])#画频谱用

    formant,loc,spec = get_formant(y,cepstL) # 返回共振峰值、共振峰位置、包络线
    freq = [i*sr/wlen for i in range(wlen2)] # 生成频率轴
    plt.subplot(2, 1, 1);plt.plot(freq, y_abs, 'k');plt.title('freq') # 画频谱
    plt.subplot(2, 1, 2);plt.plot(freq, spec, 'k');plt.title('formant estimationg') # 画共振峰
    for i in range(len(loc)):
        plt.subplot(2, 1, 2)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(spec), spec[loc[i]]], '-.k')
        plt.text(freq[loc[i]], spec[loc[i]], 'Freq={}'.format(int(freq[loc[i]])))
    plt.show()







    # F,Bw,U = Formant_Root(y,12,sr,4)
    # freq = [i * sr / 512 for i in range(256)]
    # plt.plot(freq, U)
    # plt.title('LPC求根法的共振峰估计')
    # for i in range(len(Bw)):
    #     plt.subplot(4, 1, 4)
    #     plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
    #     plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nBw={:.2f}'.format(F[i], Bw[i]))

    # plt.savefig('images/共振峰估计.png')
    # plt.close()


    # times = librosa.times_like( np.array(formant)) # 生成时间轴
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # fig, ax = plt.subplots()
    # fig
    # img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax,sr=sr)
    # ax.set(title='formant  estimation')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # ax.plot(times, np.log(loc), label='formant', color='cyan', linewidth=3)
    # ax.legend(loc='upper right')
    # plt.show()

