from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import librosa

@dataclass
class Feature:

    # Function to extract the feature e.g. from librosa
    # It takes (samples, sample rate) and returns the feature value(s)
    extractor: Callable[[np.ndarray, int], Any]
    
    # Friendly name for logging / printing
    name: str


#TODO: Not really sure this should be a constant (or kept here...)
FEATURE_MAP = {
    'rms': Feature(
        name='rms',
        extractor = lambda samples, _ : np.mean(librosa.feature.rms(y = samples))
    ),
    'pitch': Feature(
        name='pitch',
        extractor = lambda samples, sr : np.mean(librosa.pyin(y=samples, fmin=50, fmax=5000, sr=sr))
    ),
    'spectral_centroid': Feature (
        name="spectral centroid",
        extractor = lambda samples, sr : np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr))
    )
}