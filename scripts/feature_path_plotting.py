from pathlib import Path
from concatenative.audio.audio_loader import audio_loader, find_audio_files_recursively
from concatenative.analysis.corpus import Corpus
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.path.selector import generate_concatenation_path
from concatenative.visualisation.plotting import plot_feature_vs_time
from concatenative.utils.logger import setup_logger
import logging
import sys

'''
Plots a feature value over time in a ConcatenationPath
'''

ROOT = Path(__file__).resolve().parents[1]

FEATURE = 'rms'

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
    concatenation_path = generate_concatenation_path(corpus=corpus, output_length_sec=30)
    concatenated_signal = concatenation_path.render(output_length=30, output_sr=44100)
    plot_feature_vs_time(concatenated_signal, concatenation_path, feature_set[0])


