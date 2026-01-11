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

RMS = Feature (
    name='rms',
    extractor = lambda samples, _ : np.mean(librosa.feature.rms(y = samples))
)

PITCH = Feature (
    name='pitch',
    extractor = lambda samples, sr : np.mean(librosa.pyin(y=samples, fmin=50, fmax=5000, sr=sr))    
)

SPECTRAL_CENROID = Feature (
    name="spectral centroid",
    extractor = lambda samples, sr : np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr))
)


FEATURE_REGISTRY = {
    'rms': RMS,
    'pitch': PITCH,
    'spectral centroid': SPECTRAL_CENROID
}