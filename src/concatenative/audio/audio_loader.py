from .audio_snippet import AudioSnippet
from concatenative.constants import SUPPORTED_AUDIO_EXTENSIONS
from concatenative.analysis.segmentation import segment_audio
from pathlib import Path
from typing import Set, List
import numpy as np
import librosa
import logging
import random

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

    :return: List of paths to files found matching criteria provided
    '''

    if not directory_path.exists():
        raise ValueError(f"The directory path provided {directory_path} does not exist!")
    
    if not directory_path.is_dir():
        raise ValueError(f"The directory path provided {directory_path} is not a directory!")

    for extension in extensions:
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Warning, unsupported extension provided: {extension}. Must be one of {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")

    files = []
    for extension in extensions:
        logger.info(f"Searching recursively for files with {extension} extension")
        files.extend(directory_path.rglob(f"*{extension}"))

    return files


def audio_loader(path: Path, 
                 sample_rate = 44100,  
                 metadata = {}, 
                 max_clip_length = 0.1, 
                 remove_silent = True,
                 segmentation_stratgy: str = "none",
                 max_snippets: int | None = None,
                 **strategy_args) -> List[AudioSnippet]:
    '''
    Loads audio from disk into a numpy array
    
    :param path: path to the audio file on disk
    :param sample_rate: sample rate to resample the audio to (Hz)
    :param segmentation_stratgy the strategy to use to split up the audio file
    :param metadata: dictionary of extra params to store with the Snippet
    :param max_clip_length: each clip will be trimmed to this length (s)
    :param remove_silent: skip over silent audio slices if true
    :param strategy_args: extra keyword arguments for the segmentor
    :param max_snippets: Limit the number of snippets generated 
    :return: A list of AudioSnippets generated from the signal provided
    '''
    logger.info(f"Loading {path} with strategy {segmentation_stratgy} ...")

    try:
        samples, sr = librosa.load(path, sr=sample_rate)
    except Exception:
        return None
    
    target_len = int(max_clip_length * sample_rate)
    segments = segment_audio(samples=samples, sr=sr, strategy=segmentation_stratgy, **strategy_args)
    snippets = []
    for segment in segments:
        segment_samples = segment[0]

        if remove_silent and is_silent(segment_samples):
            continue

        if len(segment_samples) >= target_len:
            start = (len(segment_samples) - target_len) // 2
            segment_samples = segment_samples[start:start + target_len]

        metadata = {
            'filename': path.name,
            'start_sample': segment[1],
            'end_sample': segment[2]
        }

        snippet = AudioSnippet(
            samples=segment_samples,
            sample_rate=sr,
            metadata=metadata
        )

        snippets.append(snippet)

    if max_snippets and len(snippets) > max_snippets:
        logger.info(f"Subsampling randomly from {len(snippets)} to {max_snippets}")
        snippets = random.sample(snippets, max_snippets)

    return snippets
