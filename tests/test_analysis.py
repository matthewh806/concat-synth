from concatenative.audio import AudioSnippet
from concatenative.analysis import Corpus, analyse_snippet
from concatenative.analysis import FEATURE_MAP
from .helpers import generate_sine_wave, generate_white_noise

import pytest
from collections import deque

@pytest.fixture
def dummy_corpus():
    duration = 0.1
    sr = 44100

    return Corpus([
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.1, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'})
    ])

def test_analysed_features_match_expected(dummy_corpus):
    '''
    Tests that the features appearing in the individual AudioSnippets
    match what we expected based on the feature map used
    '''

    for snippet in dummy_corpus.snippets:
        assert(snippet.metadata)
        assert(snippet.features.keys() == FEATURE_MAP.keys())
        assert(snippet.normalised_features.keys() == FEATURE_MAP.keys())


def test_feature_normalisation(dummy_corpus):
    '''
    Tests all values are in the range [0,1] after normalisation
    Test min and max of each feature
    '''
    
    for snippet in dummy_corpus.snippets:
        for value in snippet.normalised_features.values():
            assert 0.0 <= float(value) <= 1.0

def test_values_cover_normalisation_range(dummy_corpus):
    '''
    Test that the full [0,1] range is represented by 
    normalised features
    '''

    for feature_name in FEATURE_MAP.keys():
        min_v = float('inf')
        max_v = float('-inf')
        for snippet in dummy_corpus.snippets:
            value = snippet.normalised_features[feature_name]
            min_v = min(value, min_v)
            max_v = max(value, max_v)
        
        assert min_v == pytest.approx(0.0)
        assert max_v == pytest.approx(1.0)


def test_nan_handled_correctly(dummy_corpus):
    '''
    Test that any NaN values are converted to 0.0
    '''
    
    noise_snippet = AudioSnippet(samples = generate_white_noise(0.1), sample_rate=44100, metadata={'filename': 'A'})
    _, features = analyse_snippet(noise_snippet)

    assert features['pitch'] == pytest.approx(0.0)
    