import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis

from config import Config
    
def get_mfcc(data): 
    ft1 = librosa.feature.mfcc(y=data, sr=Config.sample_rate, n_mfcc=20, fmin=0.0, fmax=500.0)
    # ft2 = librosa.feature.zero_crossing_rate(y = data)[0]
    # ft3 = librosa.feature.spectral_rolloff(y = data)[0]
    # ft4 = librosa.feature.spectral_centroid(y = data)[0]
    # ft5 = librosa.feature.spectral_contrast(y = data)[0]
    # ft6 = librosa.feature.spectral_bandwidth(y = data)[0]
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), kurtosis(ft1, axis=1), np.max(ft1, axis=1), np.median(ft1, axis=1), np.min(ft1, axis=1)))
    # ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    # ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    # ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    # ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    # ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
    # return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
    return pd.Series(np.hstack((ft1_trunc,)))