from typing import List
from .audio_snippet import AudioSnippet
from .features import FEATURE_MAP
import math
import numpy as np
import random
from collections import deque
import logging

logger = logging.getLogger(__name__)

def nearest_neighbour_search(
        snippets: List[AudioSnippet],
        target_snippet: AudioSnippet,
) -> AudioSnippet:
    '''
    Find the nearest neighbour snippet for a given target snippet

    This loops over the feature map and calculates an distance value
    between the target snippet and all other snippets.

    The snippet with the minimum distance to the target is the
    nearest neighbour
    
    :param snippets: List of AudioSnippets to check against
    :param target_snippet: AudioSnippet to use as the target
    :return: The nearnest neighbour AudioSnippet 
    '''
    
    neighbour_costs = {}

    for snippet in snippets:
        if snippet == target_snippet:
            continue
        
        distance = 0.0
        for feature_name, feature_config in FEATURE_MAP.items():
            if feature_name in snippet.normalised_features and feature_name in target_snippet.normalised_features:
                snippet_feature_value = snippet.normalised_features[feature_name]
                target_feature_value = target_snippet.normalised_features[feature_name]

                # Dont include nan values in distance calculation
                if np.isnan(snippet_feature_value) or np.isnan(target_feature_value):
                    continue

                feature_dist = feature_config.distance_fn(snippet_feature_value, target_feature_value)
                distance += feature_dist * feature_dist
                
        neighbour_costs[snippet] = math.sqrt(distance)

    neighbour = min(neighbour_costs, key=neighbour_costs.get) if len(neighbour_costs) > 0 else None

    if neighbour:
        logger.debug(f"Found neighbour for {target_snippet}: {neighbour} -  Cost: {neighbour_costs[neighbour]}")

    return neighbour


def generate_concatenation_path(snippets: List[AudioSnippet], output_length_sec: float = 10, output_sample_rate = 44100, recent_history_size = 10, cross_fade=50):
    '''
    Generates a path for the concatenator to use to create the audio collage / mosaic

    Algorithm outline
    1. From the list of snippets find a random snippet to start with
    2. Loop over the snippets until the total output length >= output_length_sec
        i. Each time around the loop finds the nearest neighbour to the target snippet
        ii. Adds the detected nearest neighbour to the concatenation path
        iv. Nearest neighbour is added to the recently_used_list to prevent immediate reuse 
        iii. Sets the detected nearest neighbour as the target for the next iteration of the loop
    3. Returns the generated list of snippets

    Note that this function uses the whole length of each snippet and so it can overshoot 
    the target `output_length_sec` param, returning a longer path. This can be trimemd 
    by the caller if necessary
    
    :param snippets: List of AudioSnippets to use as a corpus for path construction
    :param output_length_sec: Desired length of the output concatenated path
    :param output_sample_rate: Sample rate out of the output file
    :param recent_history_size: Size of the recent snippets list to exclude from re-selection
    '''

    concatenation_path = []
    target = snippets[random.randint(0, len(snippets)-1)]
    concatenation_path.append(target)

    recently_used_list = deque(maxlen= recent_history_size if recent_history_size < len(snippets) else len(snippets) // 2)
    recently_used_list.append(target.id)

    output_length = len(target.samples) / output_sample_rate

    while output_length < output_length_sec:
        searchable_snippets_pool = [
            s for s in snippets
            if s.id not in recently_used_list
        ]

        if not searchable_snippets_pool:
            logging.warning("No new snippets available to choose from. All remaining are in recently used list. Stopping early.")
            break

        target = nearest_neighbour_search(snippets=searchable_snippets_pool, target_snippet=target)

        if target is None:
            # Just randomly pick a new one in this case
            logger.warning("No new target found")
            target = snippets[random.randint(0, len(snippets)-1)]

        recently_used_list.append(target.id)

        concatenation_path.append(target)
        output_length += len(target.samples) / output_sample_rate - cross_fade / 1000

#        logger.debug(f"Running output length: {output_length}")

    logger.info(f"Generated concatenation path of length: {len(concatenation_path)} snippets. "
                f"Target length: {output_length_sec:.2f}s, "
                f"Estimated actual output {output_length:.2f}s")
    return concatenation_path