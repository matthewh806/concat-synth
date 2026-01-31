from typing import List
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def strategy_fixed(samples: np.ndarray, sr: int, segment_duration_s: float = 0.2) -> list[tuple[np.ndarray, int, int]]:
    '''
    Segment audio into fixed size slices 

    :param signal samples to segment
    :param sr sample rate of the signal
    :param segment_duration_s length of each segment in seconds
    
    :return: list of numpy segment arrays (samples, start sample, end sample)
    '''

    if len(samples) == 0:
        logger.warning("Empty signal passed into segmentor")
        return []

    segment_size = int(segment_duration_s * sr)
    if segment_size > len(samples):
        return [(samples, 0, len(samples))]
    
    num_samples = len(samples)
    segments = []
    start = 0
    end = float('-inf')
    while end < num_samples:
        end = min(start + segment_size, num_samples)
        segment = samples[start : end]
        segments.append((segment, start, end))
        start += segment_size

    return segments


def strategy_onset(samples: np.ndarray, sr: int, max_segment_length : int, hop_length: int, backtrack: bool, normalise: bool) -> list[tuple[np.ndarray, int, int]]:
    '''
    Segments audio based on detected onsets 
    :param signal samples to segment
    :param sr sample rate of the signal
    :param max_segment_length maximum length of the segment

    :return: list of numpy segment arrays (samples, start sample, end sample)

    TODO Add max length of onsets
    '''

    if len(samples) == 0:
        logger.warning("Empty signal passed into segmentor")
        return []
    
    segments = []
    onset_frames = librosa.onset.onset_detect(y=samples, sr = sr, units='frames', hop_length=hop_length, backtrack=backtrack, normalize=normalise)
    onset_samples = librosa.frames_to_samples(onset_frames)
    
    # Boundary goes from the first onset to the end of the last sample
    boundaries = np.concatenate([onset_samples, [len(samples)]])
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1] 

        if max_segment_length:
            max_segment_samples = int(max_segment_length * sr)
            duration = end - start
            end = start + max_segment_samples if duration > max_segment_samples else end

        segment = samples[start:end]
        if len(segment) > 0:
            segments.append((segment, start, end))

    return segments


def strategy_none(samples: np.ndarray, sr: int, segment_duration_s: float | None = 0.2) -> list[tuple[np.ndarray, int, int]]:
    '''
    Treat the whole signal as a single segment

    :param signal samples to segment
    :param sr sample rate of the signal
    :param segment_duration_s maximum duration of the signal to return (from 0th sample)
    
    :return: list of numpy segment arrays (samples, start sample, end sample)
    '''

    if len(samples) == 0:
        logger.warning("Empty signal passed into segmentor")
        return []
    

    if segment_duration_s:
        max_length_samples = int(segment_duration_s * sr)
        slice = samples if len(samples) < max_length_samples else samples[:max_length_samples]
        return [(slice, 0, len(slice))]
    
    return [(samples, 0, len(samples))]


SEGMENTATION_MAP = {
    'slices': strategy_fixed,
    'onsets': strategy_onset,
    'none': strategy_none
}


def segment_audio(
	samples: np.ndarray,
	sr: int,
	strategy: str,
    config: dict,
) -> List[tuple[np.ndarray, int, int]]:
    '''
    Segments the audio samples into subsets of the signal using the specified strategy
     
    :param signal samples to segment
    :param sr sample rate of the signal
    :param strategy name of the segmentation strategy to use
    :param config: dictionary containing segmentation settings

    :return: list of numpy segment arrays (samples, start sample, end sample)
    '''
      
    if strategy not in SEGMENTATION_MAP:
          raise ValueError(f"Segmentation strategy {strategy} not valid. Use one of {','.join(SEGMENTATION_MAP.keys())}")
    
    segmentation_strategy = SEGMENTATION_MAP[strategy]
    strategy_config = config.get('segmentation', {}).get(strategy, {})
    logger.info(f"Running {strategy} segmenter with parameters: {strategy_config}")
    return segmentation_strategy(samples = samples, sr = sr, **strategy_config)