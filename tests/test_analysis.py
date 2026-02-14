from concatenative.audio import AudioSnippet
from concatenative.analysis import analyse_snippet, analyse_snippets, calculate_normalised_features
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.analysis.features import Feature
from concatenative.config import load_config
from .helpers import generate_sine_wave, generate_white_noise

import pytest
import numpy as np
import librosa

@pytest.fixture
def dummy_feature_extractor():
    config = load_config()
    return FeatureExtractor([
        Feature(name="rms", extractor = lambda samples, config, _ : np.mean(librosa.feature.rms(y = samples))),
        Feature(name="pitch", extractor = lambda samples, sr, config : np.mean(librosa.pyin(y=samples, fmin=50, fmax=5000, sr=sr))),
        Feature(name="spectral centroid", extractor = lambda samples, sr, config : np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr))),
        Feature(name="mfcc", extractor = lambda samples, sr, config : np.mean(librosa.feature.mfcc(y=samples, sr = sr), axis=1))
    ], config)

@pytest.fixture
def dummy_snippets(dummy_feature_extractor):
    duration = 0.1
    sr = 44100

    return [
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.1, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'})
    ]


def test_analysed_features_match_expected(dummy_snippets, dummy_feature_extractor):
    '''
    Tests that the features appearing in the individual AudioSnippets
    match what we expected based on the feature map used
    '''

    _, features = analyse_snippet(dummy_snippets[0], dummy_feature_extractor)

    assert len(features) == len(dummy_feature_extractor)
    for feature_name in features.keys():
        assert dummy_feature_extractor.supports_feature(feature_name)


def test_feature_normalisation(dummy_snippets, dummy_feature_extractor):
    '''
    Tests all values are in the range [0,1] after normalisation
    Test min and max of each feature
    '''
    
    for snippet in dummy_snippets:
        _, features = analyse_snippet(snippet=snippet, feature_extractor=dummy_feature_extractor)
        snippet.features = features

    calculate_normalised_features(snippets=dummy_snippets, feature_extractor=dummy_feature_extractor)
    
    for snippet in dummy_snippets:
        assert len(snippet.normalised_features) == len(dummy_feature_extractor)

        for feature_name, feature_value in snippet.normalised_features.items():
            if isinstance(feature_value, np.floating):
                assert 0.0 <= float(feature_value) <= 1.0
            
            if isinstance(feature_value, list):
                for v in feature_value:
                    assert 0.0 <= float(v) <= 1.0


def test_values_cover_normalisation_range(dummy_snippets, dummy_feature_extractor):
    '''
    Test that the full [0,1] range is represented by 
    normalised features
    '''

    for snippet in dummy_snippets:
        _, features = analyse_snippet(snippet=snippet, feature_extractor=dummy_feature_extractor)
        snippet.features = features

    calculate_normalised_features(snippets=dummy_snippets, feature_extractor=dummy_feature_extractor)

    for feature in dummy_feature_extractor:
        collected = [] 
        for snippet in dummy_snippets:

            value = snippet.normalised_features[feature.name]

            if np.isscalar(value):
                collected.append(np.array([value]))
            else:
                collected.append(np.asarray(value))
            
        stacked = np.vstack(collected)
        min_vals = np.min(stacked, axis=0)
        max_vals = np.max(stacked, axis=0)
    
        assert min_vals == pytest.approx(0.0)
        assert max_vals == pytest.approx(1.0)


def test_nan_handled_correctly(dummy_feature_extractor):
    '''
    Test that any NaN values are converted to 0.0
    '''
    
    noise_snippet = AudioSnippet(samples = generate_white_noise(0.1), sample_rate=44100, metadata={'filename': 'A'})
    _, features = analyse_snippet(noise_snippet, feature_extractor=dummy_feature_extractor)

    assert features['pitch'] == pytest.approx(0.0)
    