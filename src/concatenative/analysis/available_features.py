from .features import Feature

import numpy as np
import librosa

'''
This source file lists all of the currently supported 
audio analysis and feature extraction methods from 
librosa 

These are bundled up into a dictionary called FEATURE_REGISTRY
for ease of use and lookup within the program

There's no need to use this registry or even the Feature's defined here
The system is flexible enough that Feature's can be simply defined 
in a script and passed to the system. There's no necessity to even 
use librosa as the library.
'''

def _extract_rms(samples: np.ndarray, sr: int, config: dict) -> float:
    '''
    Calculates the root mean square of a signal
    '''
    feature_config = config['features']
    return np.mean(librosa.feature.rms(y = samples, 
                                       frame_length=feature_config['frame_length'], 
                                       hop_length=feature_config['hop_length']))


def _extract_pitch(samples: np.ndarray, sr: int, config: dict) -> float:
    '''
    Calculates the pitch of a signal using the yin algorithm
    '''
    feature_config = config['features']
    return np.nanmean(librosa.pyin(y=samples,
                                   frame_length=feature_config['frame_length'],
                                   hop_length=feature_config['hop_length'],
                                   fmin=feature_config['pitch']['fmin'], 
                                   fmax=feature_config['pitch']['fmax'], 
                                   sr=sr)[0])


def _extract_spectral_centroid(samples: np.ndarray, sr: int, config: dict) -> float:
    '''
    Calculates the spectral centroid of a signal
    '''
    feature_config = config['features']
    return np.mean(librosa.feature.spectral_centroid(y=samples, 
                                                     sr=sr,
                                                     hop_length=feature_config['hop_length']))


def _extract_mfccs(samples: np.ndarray, sr: int, config: dict) -> np.ndarray:
    '''
    Calculates the mel frequency cepstral coefficients of a signal
    The librosa value returns n_mfcc coefficients per frame: (n_mfcc, num_frames)

    The return value from this method is averaged across all the frames
    so that the final output shape is (n_mfcc,)
    '''
    
    feature_config = config['features']
    mfccs = librosa.feature.mfcc(y=samples, sr = sr)
    return np.mean(mfccs, axis=1)


RMS = Feature (
    name='rms',
    extractor = _extract_rms
)

PITCH = Feature (
    name='pitch',
    extractor = _extract_pitch,
    units="Hz"
)

SPECTRAL_CENROID = Feature (
    name="spectral centroid",
    extractor = _extract_spectral_centroid,
    units="Hz"
)

MFCC = Feature (
    name="mfcc",
    extractor = _extract_mfccs
)

FEATURE_REGISTRY = {
    'rms': RMS,
    'pitch': PITCH,
    'spectral centroid': SPECTRAL_CENROID,
    'mfcc': MFCC
}