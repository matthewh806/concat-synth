import pytest
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
    assert segments == signal


