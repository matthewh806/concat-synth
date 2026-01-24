from typing import List
from collections import Counter
from concatenative.audio import AudioSnippet
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConcatenationPath:
    '''
    Class which stores a generated snippets path

    The snippets path a list of snippets which make up
    a sequence to stitch together when rendering the audio
    '''

    def __init__(
            self,
            snippets_path: List[AudioSnippet],
            cross_fade_milliseconds: float 
    ):
        self.snippets_path = snippets_path
        self.cross_fade_milliseconds = cross_fade_milliseconds


    def render(self, output_length = 10, output_sr = 44100) -> np.ndarray:
        '''
        Generates a single concatenated output file from the path data
        
        :param path: List of sample data as numpy arrays
        :param output_sr: Sample rate to save the output as (Hz)
        :param output_length: The desired output length (seconds)
        :param cross_fade: Cross fade length (milliseconds)

        :return concatenated audio as a numpy array
        '''

        logger.info("Rendering path...")

        output = self.snippets_path[0].samples.copy()
        cross_fade_samples = int((self.cross_fade_milliseconds / 1000) * output_sr)

        for snippet in self.snippets_path[1:]:
            samples = snippet.samples
            cross_fade_amount = len(samples) if len(samples) < cross_fade_samples else cross_fade_samples
            fade_out = output[-cross_fade_amount:] * np.linspace(1, 0, cross_fade_amount)
            fade_in = samples[:cross_fade_amount] * np.linspace(0, 1, cross_fade_amount)
            overlapping_region = fade_out + fade_in
            output = np.concatenate([output[:-cross_fade_amount], overlapping_region, samples[cross_fade_amount:]])

        logger.info(f"Generated a concatenated file of length {(len(output) / output_sr):.2f} seconds from {len(self.snippets_path)} samples")
        concatenation_path_length = len(output)
        target_length = int(output_length * output_sr)

        if concatenation_path_length > output_length:
            output = output[:target_length]
            logger.info(f"Trimmed final output to {output_length:.2f} seconds")
        elif concatenation_path_length < target_length:
            logger.warning(f"Final output length {(len(output) / output_sr):.2f}s is shorter than target {output_length:.2f}s")

        return output
    
    def get_stats(self):
        '''
        Retrieves some basic stats about the generated path as a string
        (Path length, number of snippets visited once, most visited)

        :return statistics string
        '''
        path_snippet_ids = [s.id for s in self.snippets_path]
        visit_counts = Counter(path_snippet_ids)

        num_visited_once = sum(1 for count in visit_counts.values() if count == 1)
        unique_snippets = len(set(visit_counts.keys()))
        most_visited = visit_counts.most_common(5)

        stats = [
            "--- Path Generation Stats ---",
            f"Path Length: {len(self)}",
            f"Unique Snippets in Path: {unique_snippets}",
            f"Snippets Visited Once: {num_visited_once}",
            f"Most Visited:"
        ]
        
        for item in most_visited:
            stats.append(f"     - ID: {str(item[0])} (Count: {item[1]})")

        return "\n".join(stats)
    
    
    def __len__(self) -> int:
        return len(self.snippets_path)
    
    def __repr__(self) -> str:
        return (f"<ConcatenationPath snippets={len(self), {self.snippets_path}}")