from typing import Tuple, List
from concatenative.analysis.corpus import Corpus
from concatenative.analysis import calculate_normalised_feature_values, analyse_snippets
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

'''
This source file provides common methods to be used
by the specific script files to avoid repeating 
boilerplate: corpus setup, path generation etc

TODO Reduce duplication from main programs concatenative_synth.py
'''

def get_parser(description="Run script"):
    '''
    Gets the basic parser that most scripts use
    Just requires a config file as input
    '''
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
    '''
    Get a list of features that are supported in the config file
    '''
    feature_set = []
    feature_names = config['features']['names']
    for feature_name in feature_names:
        if feature_name not in FEATURE_REGISTRY:
            logger.warning(f"Feature {feature_name} is not supported!")
            continue

        feature_set.append(FEATURE_REGISTRY[feature_name])

    return feature_set


def setup_corpus(config_path: str) -> Tuple[Corpus, dict]:
    '''
    Most basic setup

    1. Loads the config data into a dictionary
    2. Loads all of the snippets and performs segmentation
    3. Constructs a corpus of all the sounds (this performs analysis too)

    :return tuple containing Corpus and the config dict
    '''
    setup_logger(log_level=logging.DEBUG)
    config = load_config(Path(config_path))

    corpus_path = config['input']['corpus_path']
    file_paths = find_audio_files_recursively(Path(corpus_path))

    if len(file_paths) == 0:
        logger.error(f"No audio files found!")
        sys.exit(1)

    feature_set = get_feature_set(config)

    segmentation_strategy = config.get('segmentation', 'none').get('strategy', 'none')
    max_snippets = config.get('segmentation', None).get('max_snippets', None)
    max_clip_length = config.get('segmentation', 0.2).get('max_clip_length', 0.2)

    snippets = [
        snippet
        for file_path in file_paths
        for snippet in audio_loader(
            file_path, max_clip_length = max_clip_length, segmentation_stratgy=segmentation_strategy, max_snippets=max_snippets, config=config
        )
    ]

    return Corpus(snippets, FeatureExtractor(features=feature_set, config=config)), config


def setup_and_run_synthesis(config_path: str) -> Tuple[Corpus, ConcatenationPath, dict]:
    '''
    Does the basic setup (load audio, segment, construct corpus and analyse)
    And then generates a concatenation path based on the parameters in the config

    :return tuple containing corpus, concatenation path and the config dict
    '''
    corpus, config = setup_corpus(config_path)
    
    if 'target_path' in config['input']:
        logger.info("Generating a target based path...")
        target_path = config['input']['target_path']

        segmentation_strategy = config.get('segmentation', 'none').get('strategy', 'none')
        max_snippets = config.get('segmentation', None).get('max_snippets', None)
        max_clip_length = config.get('segmentation', 0.2).get('max_clip_length', 0.2)

        target_snippets = audio_loader(Path(target_path), 
                                       segmentation_stratgy=segmentation_strategy,
                                       max_snippets=max_snippets,
                                       max_clip_length=max_clip_length,
                                       config=config)
        
        # Extract the target features and normalise against the bounds from the corpus
        analyse_snippets(target_snippets, corpus.feature_extractor)
        for feature in get_feature_set(config):
            calculate_normalised_feature_values(target_snippets, feature.name, corpus.get_feature_bounds(feature.name))

        concat_path = generate_target_based_path(corpus, target_snippets=target_snippets)
    else:
        logger.info("Generating a freeform path...")
        concat_path = generate_freeform_path(corpus=corpus, output_length_sec=30, recent_history_size=100)

    return corpus, concat_path, config
