from typing import List
from .audio_snippet import AudioSnippet
from .features import FEATURE_MAP
import math
import numpy as np

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
