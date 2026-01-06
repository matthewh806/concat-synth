import logging
import numpy as np
from typing import List
from concatenative.utils import timed
from concatenative.audio import AudioSnippet
from concatenative.utils import run_parallel_cpu_tasks
from .features import FEATURE_MAP

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

def analyse_snippet(snippet : AudioSnippet):
    '''
    Analyses featurs for an AudioSnippet 

    The features which are calculated are defined in
    the FEATURE_MAP features.py

    This may be run in a separate process, so the features
    dict is not edited in place
    
    :param snippet: AudioSnippet to analyse
    :return snippet.id, features dict as a tuple
    '''
    samples = snippet.samples
    sample_rate = snippet.sample_rate

    features = {}
    for feature_name, feature_config in FEATURE_MAP.items():
        feature_value = feature_config.extractor(samples, sample_rate)
        # This is to prevent issues with NaN post extraction (e.g. in the kd tree construction)
        features[feature_name] = feature_value if not np.isnan(feature_value) else 0.0

    return snippet.id, features

@timed
def calculate_normalised_features(snippets : List[AudioSnippet]):
    '''
    Normalises features to be in the range [0,1]
    The calculated normal features are stored in 
    snippet.normalised_features

    The normalised_features dict is edited in place

    :param snippets: List of AudioSnippet whose features are going to be normalised
    '''

    feature_bounds = {}
    for feature_name in FEATURE_MAP.keys():
        feature_bounds[feature_name] = {'min': float('inf'), 'max': float('-inf')}

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

@timed
def analyse_snippets(snippets: List[AudioSnippet]):
    '''
    Performs a feature analysis and extraction on audio snippets
    
    :param snippets: List of AudioSnippets to analyse
    '''

    def task_complete_callback(result):
        '''
        :param result: Description
        '''
        snippet = next((snippet for snippet in snippets if snippet.id == result[0]), None)
        if snippet:
            snippet.features = result[1]
            logger.debug(f"Analysis Results for {snippet}: {log_frame_stats(features=snippet.features)}")

    logger.info(f"Starting analysis of {len(snippets)} snippets")
    run_parallel_cpu_tasks(analyse_snippet, snippets, task_complete_callback=task_complete_callback)
    calculate_normalised_features(snippets)