import logging
import numpy as np
from typing import List
from .audio_snippet import AudioSnippet

logger = logging.getLogger(__name__)

def concatenate_snippets(concatenation_path : List[AudioSnippet], output_sr = 44100, output_length = 10, cross_fade = 50):
    '''
    Generates a single concatenated output file randomly
    from the snippets provided
    
    :param snippets: List of sample data as numpy arrays
    :param output_sr: Sample rate to save the output as (Hz)
    :param output_length: The desired output length (seconds)
    :param cross_fade: Cross fade length (milliseconds)

    :return concatenated audio as a numpy array
    '''
    output = concatenation_path[0].samples.copy()
    cross_fade_samples = int((cross_fade / 1000) * output_sr)

    for snippet in concatenation_path[1:]:
        samples = snippet.samples
        cross_fade_amount = len(samples) if len(samples) < cross_fade_samples else cross_fade_samples
        fade_out = output[-cross_fade_amount:] * np.linspace(1, 0, cross_fade_amount)
        fade_in = samples[:cross_fade_amount] * np.linspace(0, 1, cross_fade_amount)
        overlapping_region = fade_out + fade_in
        output = np.concatenate([output[:-cross_fade_amount], overlapping_region, samples[cross_fade_amount:]])

    logger.info(f"Generated a concatenated file of length {(len(output) / output_sr):.2f} seconds from {len(concatenation_path)} samples")
    concatenation_path_length = len(output)
    target_length = int(output_length * output_sr)

    if concatenation_path_length > output_length:
        output = output[:target_length]
        logger.info(f"Trimmed final output to {output_length:.2f} seconds")
    elif concatenation_path_length < target_length:
        logger.warning(f"Final output length {(len(output) / output_sr):.2f}s is shorter than target {output_length:.2f}s")

    return output