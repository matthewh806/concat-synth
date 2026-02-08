from concatenative.audio.audio_loader import audio_loader, find_audio_files_recursively
from concatenative.analysis.corpus import Corpus
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.config import load_config
from concatenative.utils.logger import setup_logger
from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]

'''
This script loads all of the files in the audio_downloads directory, analyses each AudioSnippet
and performs a nearest neighbour search using a random AudioSnippet from the list
'''

if __name__ == "__main__":

    setup_logger(log_level=logging.DEBUG)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = find_audio_files_recursively(audio_dir)

    if len(file_paths) == 0:
        logging.error(f"No audio files found!")
        sys.exit(1)

    feature_set = [
        FEATURE_REGISTRY['rms'], FEATURE_REGISTRY['spectral centroid'], FEATURE_REGISTRY['pitch']
    ]

    config = load_config()
    snippets = [
        snippet
        for file_path in file_paths
        for snippet in audio_loader(
            file_path, max_clip_length = 0.2, segmentation_stratgy='slices', max_snippets=1, config = config
        )
    ]
    corpus = Corpus(snippets, FeatureExtractor(features=feature_set, config = config))
    target = corpus.get_random_snippet()

    logging.info(f"Finding nearest neighbour for target: {target}")
    
    nearest_neighbour = corpus.find_best_neighbour(
        target_snippet=target,
        exclusion_list=[]
    )

    if nearest_neighbour:
        logging.info(f"Found Nearest neighbour: {nearest_neighbour}")