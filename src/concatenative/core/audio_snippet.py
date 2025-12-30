from typing import Dict, Any
import numpy as np

class AudioSnippet:
    def __init__(
            self,
            samples: np.ndarray,
            sample_rate: int,
            metadata: Dict[str, Any] | None = None,
            features: Dict[str, Any] | None = None
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.features = {}