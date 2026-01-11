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