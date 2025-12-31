from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import librosa

@dataclass
class FeatureConfig:

    # Function to extract the feature e.g. from librosa
    # It takes (samples, sample rate) and returns the feature value(s)
    extractor: Callable[[np.ndarray, int], Any]
    
    # The function to calculate the distance between two normalised
    # feature values
    distance_fn: Callable[[Any, Any], float]
    
    # Friendly name for logging / printing
    name: str

def absolute_distance(feat_value_a, feat_value_b):
    return abs(feat_value_a - feat_value_b)


FEATURE_MAP = {
    'rms': FeatureConfig(
        name='rms',
        extractor = lambda samples, _ : np.mean(librosa.feature.rms(y = samples)),
        distance_fn = absolute_distance
    ),
    'pitch': FeatureConfig(
        name='pitch',
        extractor = lambda samples, sr : np.mean(librosa.pyin(y=samples, fmin=50, fmax=5000, sr=sr)),
        distance_fn = absolute_distance
    ),
    'spectral_centroid': FeatureConfig (
        name="spectral centroid",
        extractor = lambda samples, sr : np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr)),
        distance_fn = absolute_distance
    )
}