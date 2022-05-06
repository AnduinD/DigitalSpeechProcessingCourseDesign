import librosa,os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
 
class specshowplot:
    def __init__(self):
        self.plotlist = []
 
    def addplot(self, path, index):
        self.plotlist.insert(index, path)
 
    def show(self):
        for index, path in  enumerate(self.plotlist):
            self._show(path, index)
        plt.show()
    
    def _show(self, path, index):
        y,sr = librosa.load(path)
        count = len(self.plotlist)
        # plt.subplot(count, 2, index*2+1)
        plt.subplot(count, 1, index+1)
        # plt.title(os.path.basename(path))
        
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.5)
        # librosa.display.waveplot(y)
 
        s = librosa.stft(y)
        s[abs(s)<s.max()/5] = 0
        y = librosa.istft(s)
        tone = librosa.tone(y,sr,len(y))
        d = librosa.feature.melspectrogram( y=tone )
        d = librosa.power_to_db(d,ref=np.max)
        # s2 = plt.subplot(count, 2, index*2+2)
        librosa.display.specshow(d,x_axis='time', y_axis='mel')
 
 
if __name__ == '__main__':
    outputpathlist = ["./wavout/output_man.wav","./wavout/output_child.wav","./wavout/output_teen.wav"]
    path = "./input.wav"
    sp = specshowplot()
    sp.addplot(path, 0)
    sp.addplot(outputpathlist[0], 1)
    sp.addplot(outputpathlist[1], 2)
    sp.addplot(outputpathlist[2], 3)
    sp.show()
    