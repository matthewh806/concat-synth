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

if __name__ == "__main__":

    setup_logger(log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = find_audio_files_recursively(audio_dir)

    if len(file_paths) == 0:
        logger.error(f"No audio files found!")
        sys.exit(1)

    feature_set = [FEATURE_REGISTRY['rms']]

    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=0.2)) is not None]
    corpus = Corpus(snippets, FeatureExtractor(features=feature_set))

    plot_corpus_feature_distribution(corpus, feature_name="rms")