import librosa
import numpy as np
import wave
import math

import ctypes as ct
def nextpow2(x):
    class Float(ct.Union):
        class FloatBits(ct.Structure):
            _fields_ = [
                ('M', ct.c_uint, 23),
                ('E', ct.c_uint, 8),
                ('S', ct.c_uint, 1)
            ]
        _anonymous_ = ('bits',)
        _fields_ = [
            ('value', ct.c_float),
            ('bits', FloatBits)
        ]
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1


# 打开WAV文档
# f = wave.open("input.wav")
# 读取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)

'''谱减法降噪'''
def specsub_denoise(y,fs):
    # params = f.getparams()
    # nchannels, sampwidth, framerate, nframes = params[:4]
    # fs = framerate #取出采样率
    # str_data = f.readframes(nframes)# 读取波形数据
    # f.close()
    
    # x = np.fromstring(str_data, dtype=np.short)# 将波形数据转换为数组
    
    x = y

    # 设置分帧参数
    len_ = 20 * fs // 1000 # 帧长
    PERC = 50 #  帧移率
    stride = len_ * PERC // 100  # 帧移
    len2 = len_ - stride   # 非重叠段

    # 设置谱减参数
    Thres = 3
    Expnt = 2.0
    beta = 0.002
    G = 0.9
    
    win = np.hamming(len_)# 生成化汉明窗
    winGain = len2 / sum(win)# 窗内增益归一化 # normalization gain for overlap+add with 50% overlap

    # 用首部静态帧估计噪声幅度
    nFFT = 2 * 2 ** (nextpow2(len_))
    noise_mean = np.zeros(nFFT)

    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    # 初始化和预分配内存给变量
    k = 1;img = 1j; x_old = np.zeros(stride)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # 谱减计算处理
    for n in range(0, Nframes):
        
        insign = win * x[k-1:k + len_ - 1] # 加窗
        spec = np.fft.fft(insign, nFFT) # 计算窗内的短时频谱
        sig = abs(spec) # 取得频域幅度谱
        theta = np.angle(spec)# 取得频域相位谱
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2) # 估计信噪比

        def berouti(SNR): # 若为功率谱，生成谱减参数alpha
            if -5.0 <= SNR <= 20.0:
                a = 4 - SNR * 3 / 20
            else:
                if SNR < -5.0:
                    a = 5
                if SNR > 20:
                    a = 1
            return a

        def berouti1(SNR): # 若为幅度谱，生成谱减参数alpha
            if -5.0 <= SNR <= 20.0:
                a = 3 - SNR * 2 / 20
            else:
                if SNR < -5.0:
                    a = 4
                if SNR > 20:
                    a = 1
            return a

        if Expnt == 1.0:  # 幅度谱
            alpha = berouti1(SNRseg)
        else:  # 功率谱
            alpha = berouti(SNRseg)
        
        
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt;
        diffw = sub_speech - beta * noise_mu ** Expnt # 当纯净信号小于噪声信号的功率时
        
        def find_index(x_list): # 查找过零幅度
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list
        z = find_index(diffw)

        if len(z) > 0: # beta谱减处理
            sub_speech[z] = beta * noise_mu[z] ** Expnt # 用估计出来的噪声信号表示下限值
        if SNRseg < Thres:  # 更新对噪声谱的估计
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
            noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱

        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])# 幅度谱对称扩展
        x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta]))) # 生成输出信号复数谱  

        xi = np.fft.ifft(x_phase).real # 输出信号谱反变换
        xfinal[k-1:k + len2 - 1] = x_old + xi[0:stride] # 信号帧的滑动叠加
        x_old = xi[0 + stride:len_]
        k = k + len2

    # wf = wave.open('en_outfile.wav', 'wb')    # 保存文件
    # wf.setparams(params)# 设置参数
    # wave_data = (winGain * xfinal).astype(np.short)# 设置波形文件 .tostring()将array转换为data
    # wf.writeframes(wave_data.tostring())
    # wf.close()
    return (winGain * xfinal).astype(np.float32),len_,stride

