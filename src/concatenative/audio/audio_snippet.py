from typing import Dict, Any
import numpy as np
import uuid

class AudioSnippet:
    '''
    An AudioSnippet represents a sample subset of a signal
    The audio signal may be segmented acording to a specific rule
    (e.g. onset detection) and the resulting segments of audio
    made into AudioSnippet's
    '''
    def __init__(
            self,
            samples: np.ndarray,
            sample_rate: int,
            metadata: Dict[str, Any] | None = None,
            features: Dict[str, Any] | None = None,
            normalised_features: Dict[str, Any] | None = None
    ):
        self.id = uuid.uuid4()
        self.samples = samples
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.features = features or {}
        self.normalised_features = normalised_features or {}

    def __repr__(self):
        return f"AudioSnippet(id={self.id}, metadata={self.metadata})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, AudioSnippet):
            return NotImplemented

        return self.id == other.id
    
    def __len__(self):
        return len(self.samples)