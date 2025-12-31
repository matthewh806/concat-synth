import librosa
import numpy as np
from typing import List
from .audio_snippet import AudioSnippet
from .features import FEATURE_MAP


def analyse_snippet(snippet : AudioSnippet):
    samples = snippet.samples
    sample_rate = snippet.sample_rate

    features = {}
    for feature_name, feature_config in FEATURE_MAP.items():
        features[feature_name] = feature_config.extractor(samples, sample_rate)

    snippet.features = features


def calculate_normalised_features(snippets):
    '''
    Normalises features to be in the range [0,1]
    '''

    feature_bounds = {}
    for feature_name in FEATURE_MAP.keys():
        feature_bounds[feature_name] = {'min': float('inf'), 'max': float('-inf')}

    print(feature_bounds)

    for snippet in snippets:
        for feature_name, value in snippet.features.items():
            print(f"{feature_name}: {value:.4f}")
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
                normalised_value = (value - feat_min) / (feat_max - feat_min) 
                snippet.normalised_features[feature_name] = normalised_value

            print(f"{feature_name}: {value:.4f}, normalised: {snippet.normalised_features[feature_name]:.4f}")


def analyse_snippets(snippets: List[AudioSnippet]):
    '''
    Docstring for analyse_snippets
    
    :param snippets: Description
    :type snippets: List[AudioSnippet]
    '''
    
    for snippet in snippets:
        analyse_snippet(snippet)

    calculate_normalised_features(snippets)