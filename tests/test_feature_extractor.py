from concatenative.analysis.features import Feature
from concatenative.analysis.feature_extractor import FeatureExtractor
import pytest
import librosa
import numpy as np
from .helpers import generate_sine_wave

@pytest.fixture
def dummy_feature_extractor():
    return FeatureExtractor([
        Feature(name="rms", extractor= lambda samples, _ : np.mean(librosa.feature.rms(y = samples))),
        Feature(name="pitch", extractor = lambda samples, sr : np.mean(librosa.pyin(y=samples, fmin=50, fmax=5000, sr=sr))),
        Feature(name="spectral centroid", extractor = lambda samples, sr : np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr)))
    ])

@pytest.fixture
def dummy_signal():
    return generate_sine_wave(freq=440, duration_s=0.1, amp=0.7, sample_rate=44100)

def test_extracted_features_match_expected(dummy_feature_extractor, dummy_signal):    
    features = dummy_feature_extractor.extract(dummy_signal, 44100)
    
    assert len(features) == 3
    assert "rms" in features
    assert "pitch" in features
    assert "spectral centroid" in features

    assert type(features['rms']) == np.float32
    assert type(features['pitch']) == np.float64 # why is this float64??
    assert type(features['spectral centroid'] == np.float32)

    assert not np.isnan(features['rms'])
    assert not np.isnan(features['pitch'])
    assert not np.isnan(features['spectral centroid'])


def test_constructing_with_empty_feature_set():

    with pytest.raises(ValueError):
        feature_extractor = FeatureExtractor([])

def test_extractor_raises_for_wrong_extractor_value_type(dummy_signal):
    feature_extractor = FeatureExtractor([Feature(name="rms", extractor = lambda : 'invalid')])

    with pytest.raises(TypeError):
        feature_extractor.extract(dummy_signal, 44100)


def test_extractor_raises_for_wrong_extractor_value_type(dummy_signal):
    feature_extractor = FeatureExtractor([Feature(name="pitch", extractor = lambda samples, sr : np.mean(librosa.pyin(y=samples, sr=sr)))])

    with pytest.raises(Exception):
        feature_extractor.extract(dummy_signal, 44100)

    