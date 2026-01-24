from concatenative.utils.timer import timed
from concatenative.analysis.corpus import Corpus
from .concatenation_path import ConcatenationPath
from collections import deque
import logging

logger = logging.getLogger(__name__)

@timed 
def generate_concatenation_path(corpus: Corpus, output_length_sec: float = 10, output_sample_rate = 44100, start_snippet = None, recent_history_size = 10, cross_fade=50):
    '''
    Generates a path for the concatenator to use to create the audio collage / mosaic

    Algorithm outline
    1. From the list of snippets from the corpus to find a random snippet to start with
    2. Loop over the snippets until the total output length >= output_length_sec
        i. Each time around the loop finds the nearest neighbour to the target snippet
        ii. Adds the detected nearest neighbour to the concatenation path
        iv. Nearest neighbour is added to the recently_used_list to prevent immediate reuse 
        iii. Sets the detected nearest neighbour as the target for the next iteration of the loop
    3. Returns the generated list of snippets (ConcatenationPath instance)

    Note that this function uses the whole length of each snippet and so it can overshoot 
    the target `output_length_sec` param, returning a longer path. This can be trimemd 
    by the caller if necessary
    
    :param corpus: Corpus containing AudioSnippets and analysis data
    :param output_length_sec: Desired length of the output concatenated path
    :param output_sample_rate: Sample rate out of the output file
    :param start_snippet: Snippet to start with from the Corpus, if None a random one is picked
    :param recent_history_size: Size of the recent snippets list to exclude from re-selection
    :return ConcatenationPath containing the generated path through the snippets
    '''

    concatenation_path = []
    target = start_snippet if start_snippet and start_snippet in corpus else corpus.get_random_snippet() 
    concatenation_path.append(target)
    corpus_size = corpus.get_corpus_size()

    recently_used_list = deque(maxlen= recent_history_size if recent_history_size < corpus_size else corpus_size // 2)
    recently_used_list.append(target.id)

    output_length = len(target.samples) / output_sample_rate

    while output_length < output_length_sec:
        target = corpus.nearest_neighbour_search(target_snippet=target, exclusion_list=recently_used_list)

        if target is None:
            # Just randomly pick a new one in this case
            logger.warning("No new target found")
            target = corpus.get_random_snippet()

        recently_used_list.append(target.id)

        concatenation_path.append(target)
        output_length += len(target.samples) / output_sample_rate - cross_fade / 1000

#        logger.debug(f"Running output length: {output_length}")

    logger.info(f"Generated concatenation path of length: {len(concatenation_path)} snippets. "
                f"Target length: {output_length_sec:.2f}s, "
                f"Estimated actual output {output_length:.2f}s")
    
    return ConcatenationPath(concatenation_path, cross_fade_milliseconds=cross_fade)