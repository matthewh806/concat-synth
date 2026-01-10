from concatenative.audio import AudioSnippet
from concatenative.analysis import Corpus
from .helpers import generate_sine_wave

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


def test_feature_normalisation(dummy_corpus):
    '''
    Tests all values are in the range [0,1] after normalisation
    Test min and max of each feature
    '''
    assert(False)

def test_nan_handled_correctly(dummy_corpus):
    '''
    Test that any NaN values are converted to 0.0
    '''
    assert(False)