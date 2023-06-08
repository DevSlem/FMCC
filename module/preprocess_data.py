import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import librosa
import numpy as np
import scipy

from config import Config


def spectral_entropy(y: np.ndarray, sr: int):
    fft = np.fft.fft(y)
    p = np.abs(fft)**2
    p = p / np.sum(p, axis=0) # normalize
    entropy = -np.sum(p * np.log2(p), axis=0)
    return entropy

def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    import warnings
    warnings.filterwarnings('ignore')
    # Calculate features
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, fmin=0.0, fmax=500.0)
    mfcc_percentiles = np.percentile(mfcc, [i for i in range(0, 101, 10)], axis=1)
    mfcc_features = np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1), scipy.stats.skew(mfcc, axis=1), scipy.stats.kurtosis(mfcc, axis=1), mfcc_percentiles.flatten()))

    # Basic features
    meanfreq = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
    sd = np.std(audio_data)
    median = np.median(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))

    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    Q25, Q75 = np.percentile(spectral_contrast, [25, 75])
    IQR = Q75 - Q25

    skew = scipy.stats.skew(audio_data)
    kurt = scipy.stats.kurtosis(audio_data)

    # Advanced features
    sp_ent = np.mean(spectral_entropy(audio_data, sample_rate))
    sfm = np.mean(librosa.feature.spectral_flatness(y=audio_data))
    mode = scipy.stats.mode(audio_data)[0][0]
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))

    peakf = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0].max()

    f0 = librosa.yin(y=audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')) # type: ignore
    meanfun = np.mean(f0)
    minfun = np.min(f0)
    maxfun = np.max(f0)

    dominant = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
    meandom = np.mean(dominant)
    mindom = np.min(dominant)
    maxdom = np.max(dominant)
    dfrange = maxdom - mindom

    modindx = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate))

    warnings.filterwarnings('default')
    return np.concatenate((mfcc_features, np.array([meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm, mode, centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx])))

def preprocess_data(data_list: list[np.ndarray]) -> pd.DataFrame:
    feature_list = []
    for i in tqdm(range(len(data_list))):
        feature_list.append(extract_features(data_list[i], Config.sample_rate))
    features = np.stack(feature_list, axis=0)
    return pd.DataFrame(features)
