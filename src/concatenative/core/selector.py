from typing import List
from .audio_snippet import AudioSnippet
from .features import FEATURE_MAP
import math
import numpy as np
import random

def nearest_neighbour_search(
        snippets: List[AudioSnippet],
        target_snippet: AudioSnippet,
) -> AudioSnippet:
    
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
        print(f"Neighbour {snippet} -  Cost: {neighbour_costs[snippet]}")

    return min(neighbour_costs, key=neighbour_costs.get) if len(neighbour_costs) > 0 else None


def generate_concatenation_path(snippets: List[AudioSnippet], output_length_sec: float = 10, output_sample_rate = 44100):
    '''
    Generates a path for the concatenator to use to create the audio collage / mosaic

    Algorithm outline
    1. From the list of snippets find a random snippet to start with
    2. Loop over the snippets until the total output length >= output_length_sec
        i. Each time around the loop finds the nearest neighbour to the target snippet
        ii. Adds the detected nearest neighbour to the concatenation path
        iii. Sets the detected nearest neighbour as the target for the next iteration of the loop
    3. Returns the generated list of snippets
    
    :param snippets: List of AudioSnippets to use as a corpus for path construction
    :param output_length_sec: Desired length of the output concatenated path
    :param output_sample_rate: Sample rate out of the output file
    '''

    concatenation_path = []
    target = snippets[random.randint(0, len(snippets)-1)]
    concatenation_path.append(target)

    output_length = len(target.samples) / output_sample_rate
    while output_length <= output_length_sec:
        target = nearest_neighbour_search(snippets=snippets, target_snippet=target)

        if target is None:
            # Just randomly pick a new one in this case
            print("Warning: No new target found")
            target = snippets[random.randint(0, len(snippets)-1)]

        concatenation_path.append(target)
        output_length += len(target.samples) / output_sample_rate

    return concatenation_path