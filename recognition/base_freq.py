import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display



'''提取基音频率'''
def get_base_freq(y,sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=70, fmax=1000,sr=sr)

    # # 画图标注出每帧的基音
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    # ax.set(title='pYIN fundamental frequency estimation')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    # ax.legend(loc='upper right')
    # plt.show()
    return f0,voiced_flag,voiced_probs

if __name__ == '__main__':
    path = "./input.wav"
    y, sr = librosa.load("./input.wav")

    f0, voiced_flag, voiced_probs = get_base_freq(y,sr)

    times = librosa.times_like(f0) # 生成时间轴
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    fig
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()