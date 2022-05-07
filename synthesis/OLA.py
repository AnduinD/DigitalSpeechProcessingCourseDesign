import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
#from playsound import playsound


 
def wsola(y,sr,rate,shiftms):
    '''
    WSOLA算法语音变速重建
    param y: 待处理语音
    param sr: 原始采样率
    param rate: 规整因子a
    param shiftms: （大概像输出信号的时间分辨率？，用来确定输出插帧的间隔）
    return:
    '''
    Hs = sr * shiftms/1000 # 输出帧移长度
    f1 = Hs*2 # 帧长
    s = int(Hs)
    epstep = int(Hs * rate)  # Ha长度（输入的帧移）
    win = np.hanning(f1)
    wlen = len(y)
    wsolaed = np.zeros(int(np.floor(wlen/rate))) # 初始化处理后的音频
    sp = int(Hs * 2)  # 原信号采样帧的中心
    rp = sp + s       # 估计的最佳相似帧中心
    ep = sp + epstep  # 原信号的下一帧采样帧的中心
    outp = int(Hs)
    for i in range(outp):
        wsolaed[i] = y[i]
    data1 = np.zeros(outp)
    data2 = np.zeros(outp)
    for i in range(outp):
        data1[i] = y[sp+i]
    for i in range(outp):
        data2[i] = win[outp+i]
    #初始化
    spdata = [0 for i in range(len(data1))]
    for i in range(len(data1)):
        spdata[i] = data1[i]*data2[i]
    a = 1
    while wlen > ep + s*2:
        ref = y[rp - s +1 :rp +s]
        buff = y[ep - s*2 +1:ep + s]  # 搜索宽度
        #寻找相似区域
        corr_max = 0
        corr = 0
        corr1 = np.zeros(len(ref))
        for i in range(len(buff)-s*2): # 计算互相关序列
            compare = buff[i:i+s*2]
            for j in range(len(ref)):
                corr1[j] = ref[j]*compare[j]
            for r in range(len(corr1)):
                corr += corr1[r]
            if corr > corr_max:
                corr_max = corr
                delta = i - s
        epd = ep + delta
        #叠加(右半帧)
        data1 = np.zeros(s)
        data2 = np.zeros(s)
        for i in range(s):
            data1[i] = y[sp + i]
        for i in range(s):
            data2[i] = win[s + i]
        spdata = [0 for i in range(len(data1))]
        for i in range(len(data1)):
            spdata[i] = data1[i] * data2[i]
        #叠加（左半帧）
        data3 = np.zeros(s)
        data4 = np.zeros(s)
        for i in range(s):
            data3[i] = y[epd - s + i]
            data4[i] = win[i]
        epdata = [0 for i in range(len(data3))]
        for i in range(s):
            epdata[i] = data3[i]*data4[i]
        #叠加
        for i in range(s):
            wsolaed[outp*a + i] = spdata[i] + epdata[i]
        # 准备处理下一帧（计算起始位置）
        sp = epd
        rp = sp + s
        ep = ep + epstep
        a += 1
    return wsolaed
 
if __name__ == '__main__':
    y,sr = librosa.load('./input.wav')
    c = wsola(y,sr,1.25,10)#储存并播放变速后音频
    sf.write("D:\\pythonProject\\harvardc.wav", c, sr)#;playsound('D:\\pythonProject\\harvardc.wav')
    sr1 = round(sr*1.25)#比较重采样方式变速的音频
    sf.write("D:\\pythonProject\\harvardd.wav", y, sr1)#;playsound('D:\\pythonProject\\harvardd.wav')
    