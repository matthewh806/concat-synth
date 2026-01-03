from .audio_snippet import AudioSnippet
from typing import List
from .analysis import analyse_snippets
from .features import FEATURE_MAP
from collections import deque
import math
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)

class Corpus:
    def __init__(self, snippets: List[AudioSnippet]):
        self.snippets = snippets
        analyse_snippets(snippets)
        
    def get_random_snippet(self):
        '''
        Gets a random snippet from the corpus

        :return: A random AudioSnippet instance
        '''
        return self.snippets[random.randint(0, len(self.snippets)-1)]
    

    def get_snippet_by_id(self, id):
        '''
        Gets a snippet by ID from the corpus

        :param id: The id of the AudioSnippet to retrieve
        :return: The AudioSnippet if found, else None
        '''
        return next((s for s in self.snippets if s.id == id), None)
    

    def get_corpus_size(self):
        '''
        Gets the size of the corpus
        This is based on the number of AudioSnippet's it contains
        
        :return: The size of the corpus
        '''
        return len(self.snippets)

    
    def nearest_neighbour_search(
            self,
            target_snippet: AudioSnippet,
            exclusion_list: deque[AudioSnippet]
    ) -> AudioSnippet:
        '''
        Find the nearest neighbour snippet for a given target snippet from the corpus

        This loops over the feature map and calculates an distance value
        between the target snippet and all other snippets.

        The snippet with the minimum distance to the target is the
        nearest neighbour

        AudioSnippets in the exclusion list will be skipped in the calculation
        This can be used to prevent a list of the N most recently used samples 
        from being included
        
        :param target_snippet: AudioSnippet to use as the target
        :param a queue containing samples to skip from the nn calculation
        :return: The nearnest neighbour AudioSnippet 
        '''
        
        neighbour_costs = {}

        searchable_snippets_pool = [
            s for s in self.snippets
            if s.id not in exclusion_list
        ]

        if not searchable_snippets_pool:
            logging.warning("No new snippets available to choose from. All remaining are in recently used list. Stopping early.")
            return None

        for snippet in self.snippets:
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

        if neighbour and neighbour == target_snippet:
            logger.warning(f"Target {target_snippet} found itself as nearest neighbour!")
        elif neighbour:
            logger.debug(f"Found neighbour for {target_snippet}: {neighbour} -  Cost: {neighbour_costs[neighbour]}")

        return neighbour
