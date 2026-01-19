from typing import List
import numpy as np
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


SEGMENTATION_MAP = {
	'slices': strategy_fixed
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