from concatenative.audio import AudioSnippet
from concatenative.analysis import Corpus
from concatenative.analysis.features import Feature
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.analysis.available_features import _extract_pitch, _extract_rms, _extract_spectral_centroid
from concatenative.config import load_config
from .helpers import generate_sine_wave

import pytest
from collections import deque
import numpy as np
import librosa

@pytest.fixture
def dummy_feature_extractor():
    config = load_config()
    return FeatureExtractor([
        Feature(name="rms", extractor = _extract_rms),
        Feature(name="pitch", extractor = _extract_pitch),
        Feature(name="spectral centroid", extractor = _extract_spectral_centroid),
    ], config=config)

@pytest.fixture
def dummy_corpus(dummy_feature_extractor):
    duration = 0.1
    sr = 44100

    return Corpus([
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.1, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'})
    ], feature_extractor=dummy_feature_extractor)


def test_empty_corpus(dummy_feature_extractor):
    '''
    Test creating an empty corpus raises a value error
    '''
    with pytest.raises(ValueError):
        Corpus([], feature_extractor=dummy_feature_extractor)


def test_corpus_size(dummy_corpus):
    '''
    Test the size of the corpus is as expected
    '''
    assert(len(dummy_corpus) == 3)
    assert(dummy_corpus.get_corpus_size() == 3)


def test_corpus_contains(dummy_corpus):
    '''
    Test the corpus correctly determines if it contains a snippet
    '''
    snippet = dummy_corpus.snippets [0]
    assert(snippet in dummy_corpus)

    non_member_snippet = AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=0.5, amp=0.7), sample_rate=44100, metadata={'filename': 'D'})
    assert(non_member_snippet not in dummy_corpus)


def test_corpus_duplicates(dummy_corpus):
    '''
    Test the corpus correctly identifies duplicates
    '''
    assert(dummy_corpus.get_number_of_duplicates() == 0)

    duplicate_snippet = dummy_corpus.snippets[0]
    dummy_corpus.snippets.append(duplicate_snippet)

    assert(dummy_corpus.get_number_of_duplicates() == 1)


def test_nearest_neighbour_search(dummy_corpus):
    '''
    Test that the nearest neighbour search returns the expected snippet
    '''
    target = dummy_corpus.snippets[0]
    nearest_neighbour = dummy_corpus.nearest_neighbour_search(target, exclusion_list=deque())

    assert(nearest_neighbour != None)
    assert(nearest_neighbour.metadata['filename'] == 'B')


def test_nearest_neighbour_exclusion(dummy_corpus):
    '''
    Tests that the corpus does not select the nearest neighbour if its in the exclusion list with no fallback
    '''

    target = dummy_corpus.snippets[0]
    exclusion_list = deque([dummy_corpus.snippets[1].id])
    nearest_neighbour = dummy_corpus.nearest_neighbour_search(target, exclusion_list=exclusion_list, fallback=False)

    assert(nearest_neighbour != None)
    assert(nearest_neighbour.metadata['filename'] != 'B')
    assert(nearest_neighbour.metadata['filename'] == 'C')


def test_no_possible_neighbours(dummy_corpus):
    '''
    Tests that the corpus returns None when all other neighbours are in the exclusion list with no fallback
    '''

    target = dummy_corpus.snippets[0]
    exclusion_list = deque([s.id for s in dummy_corpus.snippets])

    nearest_neighbour = dummy_corpus.nearest_neighbour_search(target, exclusion_list=exclusion_list, fallback=False)
    assert(nearest_neighbour == None)


def test_no_possible_neighbours_fallback(dummy_corpus):
    '''
    Tests that the corpus returns None when all other neighbours are in the exclusion list with fallback
    '''

    target = dummy_corpus.snippets[0]
    exclusion_list = deque([s.id for s in dummy_corpus.snippets])

    nearest_neighbour = dummy_corpus.nearest_neighbour_search(target, exclusion_list=exclusion_list, fallback=True)
    
    assert(nearest_neighbour != None)
    assert(nearest_neighbour.metadata['filename'] == 'B')


def test_setting_nearest_neighbour_num_candidates(dummy_corpus):
    '''
    Tests that the corpus nearest neighbour method works as expected when num_candidates is specified

    Note this test is a bit weird, but makes sense, when num_candidates is just 1 it finds itself from the tree and we exclude that
    hence None is returned.
    '''

    target = dummy_corpus.snippets[0]
    exclusion_list = deque([s.id for s in dummy_corpus.snippets])

    nearest_neighbour = dummy_corpus.nearest_neighbour_search(target, exclusion_list=exclusion_list, fallback=True, num_candidates=1)
    assert(nearest_neighbour == None)


@pytest.fixture
def rms_weighted_corpus(dummy_feature_extractor):
    duration = 0.1
    sr = 44100

    feature_weights = {'rms': 0.9, 'pitch':0.1, 'spectral cenroid': 0.1}

    return Corpus([
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.1, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'})
    ], feature_extractor=dummy_feature_extractor, feature_weights=feature_weights)

@pytest.fixture
def pitch_weighted_corpus(dummy_feature_extractor):
    duration = 0.1
    sr = 44100

    feature_weights = {'rms': 0.1, 'pitch':0.9, 'spectral cenroid': 0.1}

    return Corpus([
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.1, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'})
    ], feature_extractor=dummy_feature_extractor, feature_weights=feature_weights)


@pytest.fixture
def spectral_centroid_weighted_corpus(dummy_feature_extractor):
    duration = 0.1
    sr = 44100

    feature_weights = {'rms': 0.1, 'pitch':0.1, 'spectral cenroid': 0.9}

    return Corpus([
        AudioSnippet(samples = generate_sine_wave(freq=440, duration_s=duration, amp=0.7, sample_rate=sr), sample_rate=44100, metadata={'filename': 'A'}),
        AudioSnippet(samples = generate_sine_wave(freq=450, duration_s=duration, amp=0.75, sample_rate=sr), sample_rate=44100, metadata={'filename': 'B'}),
        AudioSnippet(samples = generate_sine_wave(freq=5000, duration_s=duration, amp=0.5, sample_rate=sr), sample_rate=44100, metadata={'filename': 'C'}),
        AudioSnippet(samples = generate_sine_wave(freq=4000, duration_s=duration, amp=0.8, sample_rate=sr), sample_rate=44100, metadata={'filename': 'D'})
    ], feature_extractor=dummy_feature_extractor, feature_weights=feature_weights)



class TestWeightedDistance():
    
    def test_rms_weighted_corpus(self, rms_weighted_corpus):
        target = rms_weighted_corpus.snippets[0]
        nearest_neighbour = rms_weighted_corpus.nearest_neighbour_search(target, exclusion_list = [])
        
        assert nearest_neighbour != None
        assert nearest_neighbour.metadata['filename'] == 'B'


    def test_rms_weighted_corpus(self, pitch_weighted_corpus):
        target = pitch_weighted_corpus.snippets[1]
        nearest_neighbour = pitch_weighted_corpus.nearest_neighbour_search(target, exclusion_list = [])
    
        assert nearest_neighbour != None
        assert nearest_neighbour.metadata['filename'] == 'A'


    def test_rms_weighted_corpus(self, spectral_centroid_weighted_corpus):
        target = spectral_centroid_weighted_corpus.snippets[2]
        nearest_neighbour = spectral_centroid_weighted_corpus.nearest_neighbour_search(target, exclusion_list = [])
    
        assert nearest_neighbour != None
        assert nearest_neighbour.metadata['filename'] == 'D'

