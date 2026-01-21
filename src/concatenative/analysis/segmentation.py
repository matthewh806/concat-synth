from typing import List
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def strategy_fixed(samples: np.ndarray, sr: int, segment_duration_s: float = 0.2) -> list[np.ndarray]:
    '''
    Segment audio into fixed size slices 

    :param signal samples to segment
    :param sr sample rate of the signal
    :param segment_duration_s length of each segment in seconds
    
    :return: list of numpy segment arrays
    '''

    if len(samples) == 0:
        logger.warning("Empty signal passed into segmentor")
        return []

    segment_size = int(segment_duration_s * sr)
    if segment_size > len(samples):
        return [samples]
    
    num_samples = len(samples)
    segments = []
    start = 0
    end = float('-inf')
    while end < num_samples:
        end = min(start + segment_size, num_samples)
        segment = samples[start : end]
        segments.append(segment)
        start += segment_size

    return segments


def strategy_onset(samples: np.ndarray, sr: int, **kwargs) -> list[np.ndarray]:
    '''
    Segments audio based on detected onsets 
    :param signal samples to segment
    :param sr sample rate of the signal

    :return: list of numpy segment arrays

    TODO Add max length of onsets
    '''

    if len(samples) == 0:
        logger.warning("Empty signal passed into segmentor")
        return []
    
    segments = []
    onset_frames = librosa.onset.onset_detect(y=samples, sr = sr, units='frames')
    onset_samples = librosa.frames_to_samples(onset_frames)
    
    # Boundary goes from the first onset to the end of the last sample
    boundaries = np.concatenate([onset_samples, [len(samples)]])
    for i in range(len(boundaries) - 1):
        segment = samples[boundaries[i]:boundaries[i+1]]
        if len(segment) > 0:
            segments.append(segment)

    return segments


def strategy_none(samples: np.ndarray, sr: int, max_duration_s: float | None = 0.2) -> list[np.ndarray]:
    '''
    Treat the whole signal as a single segment

    :param signal samples to segment
    :param sr sample rate of the signal
    :param max_duration_s maximum duration of the signal to return (from 0th sample)
    
    :return: list of numpy segment arrays
    '''

    if len(samples) == 0:
        logger.warning("Empty signal passed into segmentor")
        return []
    

    if max_duration_s:
        max_length_samples = int(max_duration_s * sr)
        return [samples if len(samples) < max_length_samples else samples[:max_length_samples]]
    
    return [samples]


SEGMENTATION_MAP = {
    'slices': strategy_fixed,
    'onsets': strategy_onset,
    'none': strategy_none
}


def segment_audio(
	samples: np.ndarray,
	sr: int,
	strategy: str,
	**strategy_kwargs
) -> List[np.ndarray]:
    '''
    Segments the audio samples into subsets of the signal using the specified strategy
     
    :param signal samples to segment
    :param sr sample rate of the signal
    :param strategy name of the segmentation strategy to use
    :param strategy_kwargs keyword arguments for segmentation methods

    :return: list of numpy segment arrays
    '''
      
    if strategy not in SEGMENTATION_MAP:
          raise ValueError(f"Segmentation strategy {strategy} not valid. Use one of {','.join(SEGMENTATION_MAP.keys())}")
    
    segmentation_strategy = SEGMENTATION_MAP[strategy]
    return segmentation_strategy(samples = samples, sr = sr, **strategy_kwargs)