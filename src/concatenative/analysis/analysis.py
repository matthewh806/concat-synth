import logging
import numpy as np
from typing import List
from functools import partial
from concatenative.utils import timed
from concatenative.audio import AudioSnippet
from concatenative.utils import run_parallel_cpu_tasks
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

def log_frame_stats(features):
    '''
    Convenience method for converting the features dict
    into a nice string format for printing. 
    
    :param features: the features dict we want to print
    '''
    return {
        k: (
            f"{v.item():.3f}" 
            if isinstance(v, np.generic) and not np.isnan(v) 
            else  "nan" 
        )
        for k, v in features.items()
    }

def analyse_snippet(snippet : AudioSnippet, feature_extractor: FeatureExtractor):
    '''
    Analyses featurs for an AudioSnippet 

    The features which are calculated are defined by the feature_extractor

    This may be run in a separate process, so the features
    dict is not edited in place
    
    :param snippet: AudioSnippet to analyse
    :param feature_extractor: Instance of FeatureExtractor which determines the features to extract
    :return snippet.id, features dict as a tuple
    '''
    samples = snippet.samples
    sample_rate = snippet.sample_rate

    features = feature_extractor.extract(samples=samples, sample_rate=sample_rate)
    return snippet.id, features

@timed
def calculate_normalised_features(snippets : List[AudioSnippet], feature_extractor: FeatureExtractor):
    '''
    Normalises features to be in the range [0,1]
    The calculated normal features are stored in 
    snippet.normalised_features

    The normalised_features dict is edited in place

    :param snippets: List of AudioSnippet whose features are going to be normalised
    :param feature_extractor: Instance of FeatureExtractor which determines the features to normalise
    '''

    feature_bounds = {}
    for feature in feature_extractor:
        feature_bounds[feature.name] = {'min': float('inf'), 'max': float('-inf')}

    for snippet in snippets:
        for feature_name, value in snippet.features.items():
            if value < feature_bounds[feature_name]['min']:
                feature_bounds[feature_name]['min'] = value
            
            if value > feature_bounds[feature_name]['max']:
                feature_bounds[feature_name]['max'] = value

    for snippet in snippets:
        snippet.normalised_features = {}
        for feature_name, value in snippet.features.items():
            feat_min = feature_bounds[feature_name]['min']
            feat_max = feature_bounds[feature_name]['max']

            if np.isnan(value):
                snippet.normalised_features[feature_name] = value
            else:
                denominator = (feat_max - feat_min) 
                normalised_value = (value - feat_min) / denominator if denominator > 0 else 0
                snippet.normalised_features[feature_name] = normalised_value

            logging.debug(f"{feature_name}: {value:.4f}, normalised: {snippet.normalised_features[feature_name]:.4f}")

    return feature_bounds

@timed
def analyse_snippets(snippets: List[AudioSnippet], feature_extractor: FeatureExtractor):
    '''
    Performs a feature analysis and extraction on audio snippets
    
    :param snippets: List of AudioSnippets to analyse
    :param feature_extractor: Instance of FeatureExtractor which determines the features to extract
    '''

    def task_complete_callback(result):
        '''
        Called by run_parallel_cpu_tasks when a task completes
        :param result contains the snippet id and dictionary of features
        '''
        snippet = next((snippet for snippet in snippets if snippet.id == result[0]), None)
        if snippet:
            snippet.features = result[1]
            logger.debug(f"Analysis Results for {snippet}: {log_frame_stats(features=snippet.features)}")

    logger.info(f"Starting analysis of {len(snippets)} snippets")

    worker_task_function = partial(
        analyse_snippet,
        feature_extractor=feature_extractor
    )
    
    run_parallel_cpu_tasks(worker_task_function, snippets, task_complete_callback=task_complete_callback)


def calculate_normalised_feature_values(target_snippets: List[AudioSnippet], feature_name: str, feature_bounds: dict):
    '''
    Normalises a feature in a list of target snippets. This is based on a min / max value provided in the feature bounds 
    This is typically done to normalise a target sounds features to be within the same range as a normalised corpus of sounds

    This is written in place, into the normalised_features instance member
    
    :param target_snippets: list of snippets to normalise
    :param feature_name: name of the feature to normalise
    :param feature_bounds: dict of feature bounds, should be of the form {'min': x.yf, 'max':x.yf}
    '''
    
    logger.debug(f"Feature bounds for {feature_name}: {feature_bounds}")

    if 'min' not in feature_bounds:
        ValueError("Feature bounds dictionary is missing the min value!")

    if 'max' not in feature_bounds:
        ValueError("Feature bounds dictionary is missing the max value!")

    min = feature_bounds['min']
    max = feature_bounds['max']

    # TODO test for value float values?

    for target_snippet in target_snippets:
        feature_value = target_snippet.features[feature_name]
        target_snippet.normalised_features[feature_name] = (feature_value - min) / (max - min)