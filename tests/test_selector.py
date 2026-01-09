from concatenative.path.selector import generate_concatenation_path
from concatenative.audio import AudioSnippet
from concatenative.analysis import Corpus
from .helpers import generate_sine_wave

import pytest

@pytest.fixture
def dummy_corpus():
    duration = 0.1
    sr = 44100

    return Corpus([
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.1, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'})
    ])

def test_simple_selection(dummy_corpus):
    '''
    Tests nearest neighbour results are as expected
    '''
    output_path = generate_concatenation_path(corpus=dummy_corpus, start_snippet=dummy_corpus.snippets[0], output_length_sec=0.2, cross_fade=0)
    assert(len(output_path) == 2)

    snippets = output_path.snippets_path
    assert(snippets[0].metadata['filename'] == 'A')
    assert(snippets[1].metadata['filename'] == 'B')


