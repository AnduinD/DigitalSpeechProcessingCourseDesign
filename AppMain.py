# from sklearn import preprocessing
from turtle import left
#import __init__
import numpy as np
import scipy, librosa
import matplotlib.pyplot as plt
from recognition.denoise import specsub_denoise
from recognition.base_freq import get_base_freq
from recognition.formant import get_formant
import synthesis.voice_morph as morph
from synthesis.OLA import wsola
import visualize.specshow as visualize

 

 
if __name__ == '__main__':
    outputpath = "./wavout/output_test2.wav"
    path = "./input.wav"
    y_org, sr = librosa.load(path,sr=None)
    y_org = y_org[:50000]

    tline = np.arange(len(y_org))/sr # 生成时间轴

    # preprocessing 预处理段
    y_detrend = scipy.signal.detrend(y_org);# 去除直流信号
    y_denoise,chunk,stride = specsub_denoise(y_detrend,sr);# 谱减法去噪（通过读前几帧的内容）
   # zcr = librosa.feature.zero_crossing_rate(y_denoise)  # 过零率检测
    base_freq,voiced_flag,voiced_probs = get_base_freq(y_denoise,sr) # 基频提取
    formant,loc,spec = get_formant(y_denoise,6) # 共振峰提取--倒谱法
    #formant= librosa.lpc(y_denoise ,order=4) # 共振峰提取--LPC法

    ## 变形处理段（改基频和共振峰）
    

    y_new=0
    ## 重建段（WSOLA）

    y_out =wsola(y_new,sr,1.25,10)



    # spec = visualize.specshowplot()
    # plt.plot(formant_loc,formant,linewidth=0.05);plt.grid();
    # plt.show();

 
    # y=librosa.effects.pitch_shift(y_denoise, sr, n_steps=-1) # +3 +2 -5(移音高 即改基频)



    # y_speed =librosa.effects.time_stretch(y,3)

    '''时域可视化'''
    # plt.subplot(411);plt.plot(tline,y_org,linewidth=0.05);plt.title("org");plt.xlim(0,tline[-1]);plt.grid()
    # plt.subplot(412);plt.plot(tline,y_detrend,linewidth=0.05);plt.title("detrend");plt.xlim(0,tline[-1]);plt.grid()
    # plt.subplot(413);plt.plot(np.arange(len(y_denoise))/sr,y_denoise,linewidth=0.05);plt.title("denoise");plt.xlim(left=0);plt.grid()
    # plt.subplot(414);plt.plot(np.arange(len(y))/sr,y,linewidth=0.05);plt.title("output");plt.xlim(left=0);plt.grid()
    # plt.show()
 
    '''语谱可视化'''
    #spec = visualize.specshowplot()
    #spec.addplot(path, 0)
    #spec.addplot(outputpath, 1)
    #spec.show()

    # librosa.output.write_wav(outputpath, y, round(sr*3)) # 写wav输出


# # y = vague(6, y, sr)
# # y = pitch(6, y, sr)
# # y = speed(6, y, sr)
# # y = sample(6, y, sr)
# # y = reback(5, y, sr)
# # y = iron(6, y, sr)
# # y = quality(10, y, sr)
# # y = shrink(3, y, sr)
# # y = shrinkstep(10, y, sr)
# # y = morph.quality(20, y, sr)

# '''机器人'''
# # y = morph.speed(3, y, sr)

# # y = spread(3, y, sr)

# '''童声'''
# # y = pitch(6, y, sr)
# # y = shrinkstep(10, y, sr)