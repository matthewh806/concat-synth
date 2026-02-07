import random
import os
import sys
import math
import soundfile as sf
import logging
from pathlib import Path
from concatenative.downloaders import FreesoundAudioDownloader, YoutubeAudioDownloader, collect_snippets_parallel
from concatenative.audio.audio_loader import audio_loader, find_audio_files_recursively
from concatenative.analysis import Corpus, analyse_snippets, calculate_normalised_feature_values
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.path import generate_freeform_path, generate_target_based_path
from concatenative.constants import SUPPORTED_AUDIO_EXTENSIONS 
from concatenative.visualisation.plotting import InteractiveCorpusPlot, plot_corpus_feature_distribution, plot_feature_vs_time
from concatenative.config import load_config

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FREESOUND_API_KEY")

current_directory = Path()
download_directory = current_directory / "audio_downloads"
plots_directory = current_directory / "plots"

def get_random_phrase(word_list, phrase_len = 2):
    '''
    Given a list of words will generate random phrases of a specified length
    
    :param word_list: List of words
    :param phrase_len: Length of phrases to generate

    :return phrase string
    '''
    words = random.sample(word_list, phrase_len)
    return " ".join(words)


def load_words(filename, limit=10):
    '''
    Gets a randomised set of words from a list in a textfile
    
    :param filename: path to the list of words 
    :param limit: the number of words to fetch

    :return list of shuffled words
    '''

    with open(filename, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    random.shuffle(words)
    return words[:limit] if len(words) >= limit else words

def create_output_plots(corpus: Corpus, target_snippets, feature_set, output_signal, concatenation_path, output_dir: Path):

    if not output_dir.exists:
        logger.warning("Plotting directory {output_dir} does not exist. Skipping plotting")
        return

    # Delete existing plots in the directory
    plots_to_delete = []
    plots_to_delete.extend(output_dir.glob('*.png'))
    
    for plot in plots_to_delete:
        try:
            plot.unlink()
        except OSError as e:
            logger.warning(f"Error deleting {plot}: {e}")
    logger.info(f"Cleanup of {output_dir} complete")

    for feature in feature_set:
        if feature.name in FEATURE_REGISTRY:
            plot_corpus_feature_distribution(corpus, feature, output_dir=output_dir)
            plot_feature_vs_time(output_signal, concatenation_path, feature, output_dir=output_dir)

    if len(feature_set) == 3:
        _ = InteractiveCorpusPlot(corpus.snippets, 
                                  feature_set[0], 
                                  feature_set[1], 
                                  feature_set[2], 
                                  normalised=True,
                                  path_to_draw=concatenation_path,
                                  target_snippets=target_snippets,
                                  output_dir=output_dir)


def run_concatenator(file_paths, 
                     output_path,
                     config_path = None,
                     target_path = None,
                     feature_set = FEATURE_REGISTRY.keys(), 
                     feature_weights = {},
                     segmentation_strategy = "none",
                     output_length = 10, 
                     max_snippet_length = 0.5, 
                     max_slices_per_sample: int|None = None,
                     cross_fade = 50, 
                     plots = False):
    '''
    Loads the audio files, concatenates them and outputs the audio. 
    The type of concatenation performed depends on whether the target_path parameter is provided.
    Given this parameter a target based concatenation will be performed, otherwise a freeform path
    will be generated.

    Note: The output format is determined by the extension in the `output_path` parameter

    :param file_paths list of file paths to use in the concatenation process
    :param output_path path to the concatenated output audio file
    :param config_path path to a custom config file
    :param target_path path to a target audio file
    :param feature_set list of feature names (e.g. 'rms', 'pitch') to be used in the audio analysis
    :param segmentation_strategy the strategy for splitting up an audio sample
    :param output_length length of the output audio file
    :param max_snippet_length (s) the maximum length of each snippet
    :param cross_fade the size of the crossfade (ms) applied between each snippet in the output file

    TODO: separate this out better and rename method
    '''
    if len(file_paths) == 0:
        logger.warning(f"No audio files found!")
        sys.exit(1)

    config = load_config(Path(config_path) if config_path else None)

    if Path(output_path).suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Invalid output format {Path(output_path).suffix} provided, must be one of: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")

    for feature_name in feature_set:
        if feature_name not in FEATURE_REGISTRY:
            raise ValueError(f"Feature {feature_name} not a known feature, available features: {', '.join(FEATURE_REGISTRY.keys())}")

    logger.info(f"Loading {len(file_paths)} files into the concatenator")

    features = [FEATURE_REGISTRY[feature_name] for feature_name in feature_set]
    feature_extractor = FeatureExtractor(features=features, config=config)
    snippets = [
        snippet
        for file_path in file_paths
        for snippet in audio_loader(
            file_path, 
            max_clip_length = max_snippet_length, 
            segmentation_stratgy=segmentation_strategy, 
            max_snippets=max_slices_per_sample,
            config=config
        )
    ]
    corpus = Corpus(snippets=snippets, feature_extractor=feature_extractor, feature_weights=feature_weights)

    if target_path:
        target_snippets = audio_loader(Path(target_path), 
                                       config=config, 
                                       max_clip_length=max_snippet_length, 
                                       segmentation_stratgy=segmentation_strategy)
        
        # Extract the target features and normalise against the bounds from the corpus
        analyse_snippets(target_snippets, feature_extractor)
        for feature in features:
            calculate_normalised_feature_values(target_snippets, feature.name, corpus.get_feature_bounds(feature.name))
            
        # Generate target based concatenation path
        weight_target = config['selector']['weight_target']
        weight_previous = config['selector']['weight_previous']
        concatenation_path = generate_target_based_path(corpus=corpus, 
                                                        target_snippets = target_snippets, 
                                                        cross_fade=cross_fade, 
                                                        weight_target=weight_target, weight_previous=weight_previous)
    else:
        concatenation_path = generate_freeform_path(corpus=corpus, output_length_sec=output_length, recent_history_size=500, cross_fade=cross_fade)
    
    logger.debug(concatenation_path.get_stats())

    concatenated_audio = concatenation_path.render(output_length)
    logger.info(f"Writing to output file: {output_path}")
    sf.write(output_path, concatenated_audio, 44100)

    if plots:
        create_output_plots(corpus, 
                            target_snippets=target_snippets if target_snippets else None, 
                            feature_set=features, 
                            output_signal=concatenated_audio, 
                            concatenation_path=concatenation_path, 
                            output_dir=plots_directory)


def run_download_backend(backend_name, 
                         words_path, 
                         output_path,
                         target_path = None,
                         config_path = None,
                         feature_set = FEATURE_REGISTRY.keys(),
                         feature_weights = {},
                         segmentation_strategy = "none", 
                         output_length = 10, 
                         max_snippets = 64, 
                         max_snippet_length = 0.5,
                         max_slices_per_sample: int|None = None,
                         cross_fade = 50,
                         plots = False):
    '''
    Runs a download backend job. This name is a bit misleading as it downloads AND concatenates the audio

    The backends differ in the quality of the audio samples downloaded & hence the concatenation
    Freesound is a well structured, tagged and reliable backend. 
    YouTube on the other hand is a bit of a wildwest and we just extract random snippets of audio. 
    This actually makes YouTube much more interesting to work with as a sound source

    In the case of youtube we actually generate phrases to use based on the list of words. Each query is
    a random phrase created by combining n random words in the list. 
    In the case of freesound the individual words are passed through. This is because the search is way more
    strict in freesound & random phrases tend to yield zero results
    
    :param backend_name: The name of the backend to use for downloading audio (youtube, freesound)
    :param words_path: Path to a list of words to use as search terms for downloads
    :param output_path: Path to output the concatenated audio to
    :param target_path path to a target audio file
    :param feature_set list of feature names (e.g. 'rms', 'pitch') to be used in the audio analysis
    :param segmentation_strategy the strategy for splitting up an audio sample
    :param output_length: Desired final output length in seconds
    :param max_snippets: The maximum number of audio samples to download
    :param max_snippet_length: The maximum length of each sample when concatenating (seconds)
    :param cross_fade: Cross fade length between samples (milliseconds)
    '''

    logger.info(f"Running download backend: {backend_name}, retrieving: {max_snippets} files")

    if max_snippet_length * 1000 <= cross_fade:
        logger.error(f"Snippet length ({max_snippet_length} s) must be bigger than cross fade ({cross_fade} ms)")
        sys.exit(1)

    words = load_words(words_path, max_snippets)

    if backend_name == "youtube":
        backend = YoutubeAudioDownloader(download_directory)
        queries = [get_random_phrase(words) for _ in range(max_snippets)]
    else: # freesound
        queries = words
        results_per_word = 1 if max_snippets <= len(words) else math.ceil(len(words) / max_snippets)
        backend = FreesoundAudioDownloader(download_directory, number_of_results=results_per_word)

    download_paths = collect_snippets_parallel(backend, queries)
    run_concatenator(download_paths, 
                     output_path=output_path,
                     target_path = target_path, 
                     config_path=config_path,
                     feature_set=feature_set, 
                     feature_weights=feature_weights,
                     segmentation_strategy=segmentation_strategy, 
                     output_length=output_length, 
                     max_snippet_length=max_snippet_length, 
                     max_slices_per_sample=max_slices_per_sample,
                     cross_fade=cross_fade,
                     plots=plots)


def run_dir_backend(input_dir, 
                    output_path, 
                    target_path = None,
                    config_path = None,
                    feature_set = FEATURE_REGISTRY.keys(),
                    feature_weights = {},
                    segmentation_strategy = "none",
                    output_length = 10, 
                    max_snippet_length = 0.5, 
                    max_slices_per_sample: int|None = None,
                    cross_fade = 50, 
                    extensions = SUPPORTED_AUDIO_EXTENSIONS,
                    plots = False):
    '''
    Runs a concatenation job on a directory. The directory provided and its subdirectories are recursively
    searched and any files matching the provided extension (default mp3) will be conctatenated into a single
    output file. 
    
    :param input_dir: The directory root to use as a basis to recursively load audio files from
    :param output_path: Path to output the concatenated audio to
    :param target_path path to a target audio file
    :param feature_set list of feature names (e.g. 'rms', 'pitch') to be used in the audio analysis
    :param segmentation_strategy the strategy for splitting up an audio sample
    :param output_length: Desired final output length in seconds
    :param max_snippet_length: The maximum length of each sample when concatenating (seconds)
    :param cross_fade: Cross fade length between samples (milliseconds)
    :param extensions: The file audio file extensions to search for specifically (defaults to SUPPORTED_AUDIO_EXTENSIONS if not provided)
    '''

    if max_snippet_length * 1000 <= cross_fade:
        logger.error(f"Snippet length ({max_snippet_length} s) must be bigger than cross fade ({cross_fade} ms)")
        sys.exit(1)
    
    audio_dir = Path(input_dir)
    files = find_audio_files_recursively(audio_dir, extensions=extensions)
    run_concatenator(files, 
                     output_path=output_path,
                     target_path = target_path, 
                     config_path=config_path,
                     output_length=output_length, 
                     feature_set=feature_set, 
                     feature_weights = feature_weights,
                     segmentation_strategy=segmentation_strategy, 
                     max_snippet_length=max_snippet_length, 
                     max_slices_per_sample=max_slices_per_sample,
                     cross_fade=cross_fade,
                     plots=plots)


def main():
    '''
    This is just a basic example to demonstrate how to use the concatenative synth
    to load files directly from disk and create a single stitched output file.
    '''
    run_dir_backend(download_directory, current_directory)


if __name__=="__main__":
    main()
    
