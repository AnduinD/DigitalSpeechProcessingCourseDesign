import sys
import librosa
from . import utils

def pitch(n, y, sr):
    return librosa.effects.pitch_shift(y, sr, n_steps=n)
 
def speed(n, y, sr):
    return librosa.effects.time_stretch(y, n)
 
def sample(n, y, sr):
    return librosa.resample(y, sr, int(sr // n))
 
def reback(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool(D, size=(1, n))
    D = utils.repeat(D, n)
    return librosa.istft(D)
 
def iron(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.drop(D, n)
    return librosa.istft(D)
 
def quality(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.roll(D, n)
    return librosa.istft(D)
 
def shrink(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool(D, n, True)
    return librosa.istft(D)
 
def shrinkstep(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool_step(D, n)
    return librosa.istft(D)
 
def spread(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.spread(D, n)
    return librosa.istft(D)
 
def vague(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool(D, (1,n))
    D = utils.spread(D, (1,n))
    return librosa.istft(D)