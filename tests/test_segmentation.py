import numpy as np
from .helpers import generate_sine_wave
from concatenative.analysis.segmentation import strategy_fixed


def test_fixed_segmentation_duration_longer_than_signal():
    '''
    Tests that the whole signal is returned when the slice duration
    is longer than the signal
    '''

    sample_rate = 44100

    signal = generate_sine_wave(440, 1.0, 0.7, sample_rate)
    segments = strategy_fixed(signal, sample_rate, 2.0)

    assert len(segments) == 1
    assert (segments == signal).all()


def test_fixed_segmentation_basic_example():
    '''
    Tests that the whole signal is segmented as expected
    for the simple case where it divides perfectly
    into an equal number of chunks
    '''

    sample_rate = 44100
    segment_size_s = 0.2
    segment_size_samples = int(segment_size_s * sample_rate)

    signal = generate_sine_wave(440, 1.0, 0.7, sample_rate)
    segments = strategy_fixed(signal, sample_rate, segment_size_s)

    assert len(segments) == 5

    for idx, segment in enumerate(segments):
        assert len(segment) == segment_size_s * sample_rate
        assert (segment == signal[idx * segment_size_samples : (idx + 1) * segment_size_samples]).all()


def test_fixed_segmentation_unequal_division_example():
    '''
    Tests that the whole signal is segmented as expected
    for the simple case where it doesnt divide 
    into an equal number of chunks
    '''

    sample_rate = 44100
    segment_size_s = 0.3 # 3.33 samples
    segment_size_samples = int(segment_size_s * sample_rate)

    signal = generate_sine_wave(440, 1.0, 0.7, sample_rate)
    segments = strategy_fixed(signal, sample_rate, segment_size_s)

    assert len(segments) == 4
    assert len(segments[0]) == segment_size_samples
    assert len(segments[1]) == segment_size_samples
    assert len(segments[2]) == segment_size_samples
    assert len(segments[3]) == len(signal) - segment_size_samples*3


def test_fixed_segmentation_on_empty_signal():
    '''
    Test that for an empty signal no segments are returned
    '''

    signal = np.array([])
    segments = strategy_fixed(signal, 44100, 1.0)

    assert len(segments) == 0
