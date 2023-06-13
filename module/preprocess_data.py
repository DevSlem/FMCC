import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import librosa
import numpy as np
import scipy.stats as stats

from config import Config


def mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts MFCC features from time series audio data. 
    The features consist of MFCC, delta MFCC (velocity), and delta2 MFCC (acceleration).

    Args:
        y (ndarray): audio data `(n_samples,)`
        sr (int): sample rate

    Returns:
        mfcc_features (ndarray): `(n_features,)`
    """
    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, fmin=0, fmax=500)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    mfcc_percentiles = np.percentile(mfcc, [i for i in range(0, 101, 10)], axis=1)
    mfcc_fs = np.concatenate((
        np.mean(mfcc, axis=1), 
        np.std(mfcc, axis=1),
        mfcc_percentiles.flatten(),
        stats.skew(mfcc, axis=1), 
        stats.kurtosis(mfcc, axis=1), 
    ))

    # Then compute statistical measures as you did with original MFCCs
    delta_mfcc_fs = np.concatenate((
        np.mean(delta_mfcc, axis=1), 
        np.std(delta_mfcc, axis=1),
        np.percentile(delta_mfcc, [i for i in range(0, 101, 10)], axis=1).flatten(),
        stats.skew(delta_mfcc, axis=1), 
        stats.kurtosis(delta_mfcc, axis=1), 
    ))

    delta2_mfcc_fs = np.concatenate((
        np.mean(delta2_mfcc, axis=1), 
        np.std(delta2_mfcc, axis=1),
        np.percentile(delta2_mfcc, [i for i in range(0, 101, 10)], axis=1).flatten(),
        stats.skew(delta2_mfcc, axis=1), 
        stats.kurtosis(delta2_mfcc, axis=1), 
    ))

    return np.concatenate((
        mfcc_fs,
        delta_mfcc_fs,
        delta2_mfcc_fs,
    ))

def spectral_entropy(y: np.ndarray, sr: int):
    fft = np.fft.fft(y)
    p = np.abs(fft)**2
    p = p / np.sum(p, axis=0) # normalize
    entropy = -np.sum(p * np.log2(p), axis=0)
    return entropy

def spectral_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts spectral features from time series audio data.

    Args:
        y (ndarray): audio data `(n_samples,)`
        sr (int): sample rate

    Returns:
        spectral_features (ndarray): `(n_features,)`
    """
    # spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    freq_mean = np.mean(spec_cent)
    # freq_std = np.std(audio_data)
    freq_std = np.std(spec_cent)
    freq_median = np.median(spec_cent)

    # spectral contrast
    spec_cont = librosa.feature.spectral_contrast(y=y, sr=sr)
    Q25, Q75 = np.percentile(spec_cont, [25, 75])
    IQR = Q75 - Q25
    
    # spectral entropy
    sp_ent = np.mean(spectral_entropy(y, sr))
    
    # spectral flatness
    sfm = np.mean(librosa.feature.spectral_flatness(y=y))
    mode = stats.mode(y)[0][0]
    # centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))

    # spectral bandwidth
    dominant = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    peakf = dominant[0].max()
    meandom = np.mean(dominant)
    mindom = np.min(dominant)
    maxdom = np.max(dominant)
    dfrange = maxdom - mindom

    modindx = np.mean(spec_cont)

    return np.array([
        freq_mean, 
        freq_std, 
        freq_median, 
        Q25, 
        Q75, 
        IQR,
        sp_ent, 
        sfm, 
        mode, 
        peakf,
        meandom, 
        mindom, 
        maxdom, 
        dfrange, 
        modindx
    ])
    
def audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts audio features from time series audio data.

    Args:
        y (ndarray): audio data `(n_samples,)`
        sr (int): sample rate

    Returns:
        audio_features (ndarray): `(n_features,)`
    """
    audio_skew = stats.skew(y)
    audio_kurt = stats.kurtosis(y)
    
    f0 = librosa.yin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')) # type: ignore
    meanfun = np.mean(f0)
    minfun = np.min(f0)
    maxfun = np.max(f0)
    
    return np.array((
        audio_skew, 
        audio_kurt,
        meanfun,
        minfun,
        maxfun,
    ))

def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    import warnings
    warnings.filterwarnings('ignore')
    
    mfcc_fs = mfcc_features(audio_data, sample_rate)
    spectral_fs = spectral_features(audio_data, sample_rate)
    audio_fs = audio_features(audio_data, sample_rate)

    warnings.filterwarnings('default')
    return np.concatenate((
        mfcc_fs,
        spectral_fs,
        audio_fs,
    ))

def preprocess_data(data_list: list[np.ndarray]) -> pd.DataFrame:
    feature_list = []
    for i in tqdm(range(len(data_list))):
        feature_list.append(extract_features(data_list[i], Config.sample_rate))
    features = np.stack(feature_list, axis=0)
    return pd.DataFrame(features)
