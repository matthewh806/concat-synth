from .audio_snippet import AudioSnippet
from concatenative.constants import SUPPORTED_AUDIO_EXTENSIONS
from pathlib import Path
from typing import Set
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def is_silent(samples, rms_threshold=1e-4):
    '''
    Determines if a an audio array is silent (coarse)
    This is done by calculating the RMS value and checking
    if this is above / below a threshold value
    
    :param samples: numpy array of samples
    :param rms_threshold: threshold below which we consider silence

    :return: true if rms is less than threshold (silence), false otherwise
    '''
    rms = np.sqrt(np.mean(samples**2))
    return rms < rms_threshold

def find_audio_files_recursively(directory_path: Path, extensions: Set[str] = SUPPORTED_AUDIO_EXTENSIONS):
    '''
    Recursively searches the directory provided for audio files matching the provided extensions
    
    :param directory_path: Path of root directory to search
    :param extensions: Set of supported file extensions. Must include the . (e.g. {'.mp3'})
    '''
    for extension in extensions:
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Warning, unsupported extension provided: {extension}. Must be one of {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")

    files = []
    for extension in extensions:
        logger.info(f"Searching recursively for files with {extension} extension")
        files.extend(directory_path.rglob(f"*{extension}"))

    return files


def audio_loader(path: Path, sample_rate = 44100, metadata = {}, max_clip_length = 0.1, remove_silent = True) -> AudioSnippet:
    '''
    Loads audio from disk into a numpy array
    
    :param path: path to the audio file on disk
    :param sample_rate: sample rate to resample the audio to (Hz)
    :param metadata: dictionary of extra params to store with the Snippet
    :param max_clip_length: the clip will be trimmed to this length (s)
    :param remove_silent: skip over silent audio slices if true
    :return: AudioSnippet instance containing samples in a 1d numpy array
    '''
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
    
    if len(metadata) == 0:
        metadata = {
            'filename': path.name
        }

    return AudioSnippet(
        samples=samples,
        sample_rate=sr,
        metadata=metadata
    )
