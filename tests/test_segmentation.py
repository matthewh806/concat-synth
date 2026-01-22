import numpy as np
from .helpers import generate_sine_wave, generate_click_track
from concatenative.analysis.segmentation import strategy_fixed, strategy_onset, strategy_none


class TestFixedSegmentation():

    def test_empty_signal(self):
        '''
        Test that for an empty signal no segments are returned
        '''

        signal = np.array([])
        segments = strategy_fixed(signal, 44100, 1.0)

        assert len(segments) == 0


    def test_duration_longer_than_signal(self):
        '''
        Tests that the whole signal is returned when the slice duration
        is longer than the signal
        '''

        sample_rate = 44100

        signal = generate_sine_wave(440, 1.0, 0.7, sample_rate)
        segments = strategy_fixed(signal, sample_rate, 2.0)

        assert len(segments) == 1
        assert (segments[0][0] == signal).all()


    def test_basic_example(self):
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
            segment_samples = segment[0]
            assert len(segment_samples) == segment_size_s * sample_rate
            assert (segment_samples == signal[idx * segment_size_samples : (idx + 1) * segment_size_samples]).all()


    def test_unequal_division_example(self):
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
        assert len(segments[0][0]) == segment_size_samples
        assert len(segments[1][0]) == segment_size_samples
        assert len(segments[2][0]) == segment_size_samples
        assert len(segments[3][0]) == len(signal) - segment_size_samples*3


class TestOnsetSegmentation():
        
        def test_empty_signal(self):
            '''
            Test that for an empty signal no segments are returned
            '''

            signal = np.array([])
            segments = strategy_onset(signal, 44100)

            assert len(segments) == 0


        def test_signal_with_no_onsets(self):
            '''
            Test that a signal with no onsets no segments are returned
            '''

            signal = np.zeros(44100)
            segments = strategy_onset(signal, 44100)

            assert len(segments) == 0


        def test_predictable_signal(self):
            '''
            Tests on a simple, predictable click track
            '''

            click_track = generate_click_track(1, [0.25, 0.5, 0.8], sample_rate=44100)
            segments = strategy_onset(click_track, 44100)

            assert len(segments) == 3


class TestNoneSegmentation():

        def test_empty_signal(self):
            '''
            Test that for an empty signal no segments are returned
            '''

            signal = np.array([])
            segments = strategy_none(signal, 44100)

            assert len(segments) == 0

        
        def test_full_signal_returned(self):
            '''
            Test that a full signal is returned when None is given as max duration
            '''

            signal = generate_sine_wave(440, 1.0, 0.7, 44100)
            segments = strategy_none(signal, 44100, max_duration_s=None)

            assert len(segments) == 1
            assert (segments[0][0] == signal).all()

        
        def test_sub_signal_returned(self):
            '''
            Test that subset of the signal is returned when a max_duration is provided
            '''

            sample_rate = 44100
            signal = generate_sine_wave(440, 1.0, 0.7, sample_rate=sample_rate)
            segments = strategy_none(signal, sr=sample_rate, max_duration_s=0.5)

            assert len(segments) == 1
            assert len(segments[0][0]) == int(0.5 * sample_rate)
            assert (segments[0][0] == signal[:int(0.5 * sample_rate)]).all()

        
        def test_duration_longer_than_signal(self):
            '''
            Test that the whole signal is returned when max duration is longer than the signal
            '''

            sample_rate = 44100
            signal = generate_sine_wave(440, 1.0, 0.7, sample_rate=sample_rate)
            segments = strategy_none(signal, sr=sample_rate, max_duration_s=2.0)

            assert len(segments) == 1
            assert len(segments[0][0]) == len(signal)
            assert (segments[0][0] == signal).all()