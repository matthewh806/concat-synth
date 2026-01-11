import numpy as np
from .features import Feature
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:

    def __init__(self, features: List[Feature]):
        '''
        Initialises the FeatureExtractor instance with a list of 
        features to be extracted from audio samples
        '''

        if len(features) == 0:
            raise ValueError("Constructring FeatureExtractor with empty feature list")

        self.features = features


    def extract(self, samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
        '''
        Extracts all of the desired features for a given audio signal

        TODO Currently only supports returning a single value for the feature 
        But this doesn't always make sense, e.g. mel coefficients.

        :param samples: The numpy array of the audio samples
        :param sample_rate: The sample rate of the audio

        :return: A dictionary of feature names to the calculated values
        '''

        features = {}
        for feature in self.features:
            try:
                feature_value = feature.extractor(samples, sample_rate)

                if not isinstance(feature_value, np.floating):
                    raise TypeError(f"The extracted feature value {feature_value} is expected to be a numpy float not a {type(feature_value)}")

                # This is to prevent issues with NaN post extraction (e.g. in the kd tree construction)
                features[feature.name] = feature_value if not np.isnan(feature_value) else 0.0
            except Exception as e:
                logger.error(f"Error extracting feature {feature.name}: {e}")
                raise


        return features