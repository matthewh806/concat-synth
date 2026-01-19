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

    return []