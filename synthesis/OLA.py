import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
#from playsound import playsound

y,sr = librosa.load('./input.wav')
 
def wsola(y,sr,rate,shiftm):
    Hs = sr * shiftm/1000
    f1 = Hs*2
    s = int(Hs)
    epstep = int(Hs * rate)
    win = np.hanning(f1)
    wlen = len(y)
    wsolaed = np.zeros(int(np.floor(wlen/rate)))
    sp = int(Hs * 2)
    rp = sp + s
    ep = sp + epstep
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
        buff = y[ep - s*2 +1:ep + s]
        #寻找相似区域
        corr_max = 0
        corr = 0
        corr1 = np.zeros(len(ref))
        for i in range(len(buff)-s*2):
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
        sp = epd
        rp = sp + s
        ep = ep + epstep
        a += 1
    return wsolaed
 
#储存并播放变速后音频
c = wsola(y,sr,1.25,10)
sf.write("D:\\pythonProject\\harvardc.wav", c, sr)
#playsound('D:\\pythonProject\\harvardc.wav')
 
#比较重采样方式变速的音频
sr1 = round(sr*1.25)
sf.write("D:\\pythonProject\\harvardd.wav", y, sr1)
#playsound('D:\\pythonProject\\harvardd.wav')