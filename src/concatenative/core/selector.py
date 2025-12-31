from typing import List
from .audio_snippet import AudioSnippet
import math

def nearest_neighbour_search(
        snippets: List[AudioSnippet],
        target_snippet: AudioSnippet,
        feature_names: List[str]
) -> AudioSnippet:
    
    neighbour_costs = {}

    for snippet in snippets:
        if snippet == target_snippet:
            continue
        
        distance = 0.0
        for feature_name in feature_names:
            if feature_name in snippet.normalised_features and feature_name in target_snippet.normalised_features:
                snippet_feature_value = snippet.normalised_features[feature_name]
                target_feature_value = target_snippet.normalised_features[feature_name]
                distance += (snippet_feature_value - target_feature_value) * (snippet_feature_value - target_feature_value)
        
        neighbour_costs[snippet] = math.sqrt(distance)
        print(f"Neightbour Cost: {neighbour_costs[snippet]}")

    return min(neighbour_costs, key=neighbour_costs.get) if len(neighbour_costs) > 0 else None
