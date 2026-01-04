from .audio_snippet import AudioSnippet
from typing import List, Dict, Optional
from .analysis import analyse_snippets
from .features import FEATURE_MAP
from collections import deque
import math
import logging
import numpy as np
import random
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

class Corpus:
    def __init__(self, snippets: List[AudioSnippet]):

        if not snippets:
            ValueError("Cannot initialise an empty corpus.")

        self.snippets = snippets

        # The stored feature search space tree
        self.search_tree: Optional[KDTree] = None

        # A mapping from the internal KDTree index back to the snippet object
        self.index_to_snippet_map : Dict[int, AudioSnippet] = {}

        analyse_snippets(snippets)
        self.build_feature_space()
        
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
    

    def _get_snippet_feature_vector(self, snippet : AudioSnippet):
        '''
        Getst the k dimensional feature vector for an AudioSnippet
        This will look through the feature map and for each feature
        add a dimension to the vector
        
        :param snippet: The snippet to generate the vector for
        '''
        
        feature_vector = []

        for feature_name, _ in FEATURE_MAP.items():
            if feature_name in snippet.normalised_features:
                value = snippet.normalised_features[feature_name]

                if value is None:
                    logger.warning(f"Snippet {snippet.id} is missing the feature '{feature_name}' and will be excluded from the feature space")
                    return None
                
                feature_vector.append(value)

        return np.array(feature_vector)
    
    
    def build_feature_space(self):
        '''
        Builds a k dimensional feature space tree, this is stored for quickly 
        finding nearest neighbours.

        The KDTree implementation used here is from scipy.signal which simply uses
        a euclidean distance measure.
        '''

        feature_vectors = []
        for snippet in self.snippets:
            vector = self._get_snippet_feature_vector(snippet)
            if vector is not None:
                map_index = len(feature_vectors)
                self.index_to_snippet_map[map_index] = snippet
                feature_vectors.append(vector)

        self.search_tree = KDTree(np.array(feature_vectors))

    
    def nearest_neighbour_search(
            self,
            target_snippet: AudioSnippet,
            exclusion_list: deque[AudioSnippet],
            num_candidates: int = 10
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

        target_feature_vector = self._get_snippet_feature_vector(target_snippet)
        #num_neighbours = num_candidates if num_candidates

        try:
            distances, indices = self.search_tree.query(target_feature_vector, k = num_candidates)
        except Exception as e:
            logger.error(f"KDTree query failed: {e}")
            return None

        for index, distance in zip(indices, distances):
            candidate_snippet = self.index_to_snippet_map[index]

            if candidate_snippet != target_snippet and candidate_snippet not in exclusion_list:
                logger.debug(f"Found neighbour for {target_snippet}: {candidate_snippet} -  Distance: {distance}")
                return candidate_snippet
            
        # Fallback in case we didn't find a neighbour
        if indices.size > 0:
            candidate_snippet = self.index_to_snippet_map[0]
            logger.debug(f"All nearest neighbours recently used, fallback for {target_snippet}: {candidate_snippet} -  Distance: {distances[0]}")

        logger.error(f"No suitable neighbours found for {target_snippet}.")
        return None
