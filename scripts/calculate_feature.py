from pathlib import Path
from concatenative.utils.logger import setup_logger
from concatenative.audio.audio_loader import audio_loader
from concatenative.analysis.available_features import FEATURE_REGISTRY
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]

FEATURE = 'pitch'

if __name__ == "__main__":
    setup_logger(log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    audio_dir = Path(ROOT / "audio_downloads")
    audio_path = audio_dir / "Aurorae Aurorae__d24c7a/yX8bYwl9rKc.wav"

    if FEATURE not in FEATURE_REGISTRY:
        logger.error(f"Feature {FEATURE} is not supported!")
        sys.exit(1)

    snippet = audio_loader(audio_path, sample_rate=44100, max_clip_length=0.2)

    feature = FEATURE_REGISTRY[FEATURE]

    print(f"Num samples: {len(snippet.samples)}")
    print(f"Feature result: {feature.extractor(snippet.samples, snippet.sample_rate)}")