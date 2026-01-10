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


def test_empty_corpus():
    '''
    Test creating an empty corpus raises a value error
    '''
    with pytest.raises(ValueError):
        Corpus([])


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