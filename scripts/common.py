from typing import Tuple, List
from concatenative.analysis.corpus import Corpus
from concatenative.path.concatenation_path import ConcatenationPath
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.analysis.features import Feature
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.config import load_config
from concatenative.utils.logger import setup_logger
from concatenative.audio.audio_loader import find_audio_files_recursively, audio_loader
from concatenative.path import generate_freeform_path, generate_target_based_path
import logging
import argparse
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def get_parser(description="Run script"):
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--config", type = str, default="scripts/configs/default_config.toml",
        help=(
            "Provide a custom config file for tweaking under the hood settings.\n"
            "See 'configs/default_config.toml' in the for the default example"
        )
    )

    return parser

def get_feature_set(config: dict) -> List[Feature]:
    feature_set = []
    feature_names = config['features']['names']
    for feature_name in feature_names:
        if feature_name not in FEATURE_REGISTRY:
            logger.warning(f"Feature {feature_name} is not supported!")
            continue

        feature_set.append(FEATURE_REGISTRY[feature_name])

    return feature_set

def setup_corpus(config_path: str) -> Tuple[Corpus, dict]:

    setup_logger(log_level=logging.DEBUG)
    config = load_config(Path(config_path))

    corpus_path = config['input']['corpus_path']
    file_paths = find_audio_files_recursively(Path(corpus_path))

    if len(file_paths) == 0:
        logger.error(f"No audio files found!")
        sys.exit(1)

    feature_set = get_feature_set(config)

    snippets = [
        snippet
        for file_path in file_paths
        for snippet in audio_loader(
            file_path, max_clip_length = 0.2, segmentation_stratgy='slices', max_snippets=1, config=config
        )
    ]

    return Corpus(snippets, FeatureExtractor(features=feature_set, config=config)), config


def setup_and_run_synthesis(config_path: str) -> Tuple[Corpus, ConcatenationPath, dict]:
    corpus, config = setup_corpus(config_path)
    # TODO check if its a freeform or target based path

    return corpus, generate_freeform_path(corpus=corpus, output_length_sec=30), config
