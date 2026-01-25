from concatenative.audio import AudioSnippet
from .analysis import analyse_snippets
from .feature_extractor import FeatureExtractor
from .available_features import FEATURE_REGISTRY
from collections import deque
from typing import List, Dict, Optional
import uuid
import logging
import numpy as np
import random
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

class Corpus:
    def __init__(self, snippets: List[AudioSnippet], feature_extractor: FeatureExtractor, feature_weights: Dict[str, float] = {}):

        if not snippets:
            ValueError("Cannot initialise an empty corpus.")

        self.snippets = snippets
        self.feature_extractor = feature_extractor

        self.feature_weights = feature_weights

        # The stored feature search space tree
        self.search_tree: Optional[KDTree] = None

        # A mapping from the internal KDTree index back to the snippet object
        self.index_to_snippet_map : Dict[int, AudioSnippet] = {}

        analyse_snippets(snippets, feature_extractor=feature_extractor)

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
        return len(self)
    
    def get_number_of_duplicates(self):
        '''
        Gets the number of duplicates snippets in the corpus
        
        :return: The number of duplicates
        '''

        unique_filenames = {
            filename for snippet in self.snippets 
            if (filename := snippet.metadata.get('filename'))
        }

        return self.get_corpus_size() - len(unique_filenames)

    def _get_snippet_feature_vector(self, snippet : AudioSnippet, feature_extractor: FeatureExtractor):
        '''
        Getst the k dimensional feature vector for an AudioSnippet
        This will look through the feature map and for each feature
        add a dimension to the vector
        
        :param snippet: The snippet to generate the vector for
        '''
        
        feature_vector = []

        for feature in feature_extractor:
            if feature.name in snippet.normalised_features:
                value = snippet.normalised_features[feature.name]

                if value is None:
                    logger.warning(f"Snippet {snippet.id} is missing the feature '{feature.name}' and will be excluded from the feature space")
                    return None
                
                # Use the supplied weight if available
                weight = self.feature_weights[feature.name] if feature.name in self.feature_weights else feature.default_weight
                feature_vector.append(np.sqrt(weight) * value)

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
            vector = self._get_snippet_feature_vector(snippet, self.feature_extractor)
            if vector is not None:
                map_index = len(feature_vectors)
                self.index_to_snippet_map[map_index] = snippet
                feature_vectors.append(vector)

        if not feature_vectors:
            raise ValueError(
                "Cannot build feature space for an empty corpus."
                "Ensure corpus has snippets with valid feature vectors"
            )

        self.search_tree = KDTree(np.array(feature_vectors))

    
    def nearest_neighbour_search(
            self,
            target_snippet: AudioSnippet,
            exclusion_list: deque[uuid.UUID],
            num_candidates: int = 10,
            fallback: bool = True
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
        :param if True if no neighbours are found try to fallback on the best alternative option, False returns none if none found
        :return: The nearnest neighbour AudioSnippet 
        '''

        target_feature_vector = self._get_snippet_feature_vector(target_snippet, self.feature_extractor)
        candidates_to_search = min(len(self), num_candidates)

        try:
            distances, indices = self.search_tree.query(target_feature_vector, k = candidates_to_search)
        except Exception as e:
            logger.error(f"KDTree query failed: {e}")
            return None
        
        if num_candidates == 1:
            indices = [indices]
            distances = [distances]

        for index, distance in zip(indices, distances):
            candidate_snippet = self.index_to_snippet_map[int(index)]

            if candidate_snippet != target_snippet and candidate_snippet.id not in exclusion_list:
                logger.debug(f"Found neighbour for {target_snippet}: {candidate_snippet} -  Distance: {distance}")
                return candidate_snippet
            
        # Fallback in case we didn't find a neighbour
        if fallback and len(indices) > 1:
            # Note used 1 here because a tree will find itself if queried with its own feature vector
            # If the idea of targets is introduced I'll need to be more careful here!
            candidate_snippet = self.index_to_snippet_map[1]
            logger.debug(f"All nearest neighbours recently used, fallback for {target_snippet}: {candidate_snippet} -  Distance: {distances[1]}")
            return candidate_snippet

        logger.error(f"No suitable neighbours found for {target_snippet}.")
        return None

    def __len__(self) -> int:
        return len(self.snippets)
    
    def __contains__(self, snippet) -> bool:
        return any(s.id == snippet.id for s in self.snippets)
    
    def __repr__(self) -> str:
        return (f"<Corpus snippets={len(self)}>")