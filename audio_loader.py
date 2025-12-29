from typing import Dict, Any
from pathlib import Path
import numpy as np
import librosa

class AudioSnippet:
    def __init__(
            self,
            samples: np.ndarray,
            sample_rate: int,
            metadata: Dict[str, Any] | None = None,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.metadata = metadata or {}

def is_silent(samples, rms_threshold=1e-4):
    rms = np.sqrt(np.mean(samples**2))
    return rms < rms_threshold


def audio_loader(path: Path, sample_rate = 44100, metadata = {}, max_clip_length = 0.1, remove_silent = True) -> AudioSnippet:
    try:
        samples, sr = librosa.load(path, sr=sample_rate)
    except Exception:
        return None
    
    # Some backends don't always always return the exact length of audio requested
    # So trim manually
    target_len = int(max_clip_length * sample_rate)
    if len(samples) >= target_len:
        start = (len(samples) - target_len) // 2
        samples = samples[start:start + target_len]

    if is_silent(samples):
        return None

    return AudioSnippet(
        samples=samples,
        sample_rate=sr,
        metadata=metadata
    )
