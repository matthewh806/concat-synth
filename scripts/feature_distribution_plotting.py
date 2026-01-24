from pathlib import Path
from concatenative.audio.audio_loader import audio_loader, find_audio_files_recursively
from concatenative.analysis.corpus import Corpus
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.visualisation.plotting import plot_corpus_feature_distribution
from concatenative.utils.logger import setup_logger
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]

FEATURE = 'pitch'

if __name__ == "__main__":

    setup_logger(log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = find_audio_files_recursively(audio_dir)

    if len(file_paths) == 0:
        logger.error(f"No audio files found!")
        sys.exit(1)

    if FEATURE not in FEATURE_REGISTRY:
        logger.error(f"Feature {FEATURE} is not supported!")
        sys.exit(1)

    feature_set = [FEATURE_REGISTRY[FEATURE]]
    snippets = [
        snippet
        for file_path in file_paths
        for snippet in audio_loader(
            file_path, max_clip_length = 0.2, segmentation_stratgy='slices', segment_duration_s=1.0, max_snippets=1
        )
    ]
    corpus = Corpus(snippets, FeatureExtractor(features=feature_set))
    plot_corpus_feature_distribution(corpus, feature=feature_set[0], bins=100)