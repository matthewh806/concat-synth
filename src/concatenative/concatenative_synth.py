import random
import os
import sys
import math
import soundfile as sf
import logging
from pathlib import Path
from concatenative.downloaders import FreesoundAudioDownloader, YoutubeAudioDownloader, collect_snippets_parallel
from concatenative.audio import audio_loader
from concatenative.analysis import Corpus
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.path import generate_concatenation_path

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FREESOUND_API_KEY")

current_directory = Path()
download_directory = current_directory / "audio_downloads"


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


def run_concatenator(file_paths, output_path, feature_set, output_length = 10, max_snippet_length = 0.5, cross_fade = 50):
    '''
    Loads the audio files, concatenates them and outputs the audio 

    :param file_paths list of file paths to use in the concatenation process
    :param output_path path to the concatenated output audio file
    :param feature_set list of feature names (e.g. 'rms', 'pitch') to be used in the audio analysis
    :param output_length length of the output audio file
    :param max_snippet_length (s) the maximum length of each snippet
    :param cross_fade the size of the crossfade (ms) applied between each snippet in the output file

    TODO: separate this out better and rename method
    '''
    if len(file_paths) == 0:
        logger.warning(f"No audio files found!")
        sys.exit(1)

    for feature_name in feature_set:
        if feature_name not in FEATURE_REGISTRY:
            raise ValueError(f"Feature {feature_name} not a known feature, available features: {', '.join(FEATURE_REGISTRY.keys())}")

    logger.info(f"Loading {len(file_paths)} files into the concatenator")

    features = [FEATURE_REGISTRY[feature_name] for feature_name in feature_set]
    feature_extractor = FeatureExtractor(features=features)
    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=max_snippet_length)) is not None]
    corpus = Corpus(snippets=snippets, feature_extractor=feature_extractor)
    concatenation_path = generate_concatenation_path(corpus=corpus, output_length_sec=output_length, cross_fade=cross_fade)
    logger.debug(concatenation_path.get_stats())

    concatenated_audio = concatenation_path.render(output_length)
    sf.write(output_path, concatenated_audio, 44100)


def run_download_backend(backend_name, words_path, output_path, feature_set, output_length = 10, max_snippets = 64, max_snippet_length = 0.5, cross_fade = 50):
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
    :param feature_set list of feature names (e.g. 'rms', 'pitch') to be used in the audio analysis
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
    run_concatenator(download_paths, output_path=output_path, feature_set=feature_set, output_length=output_length, max_snippet_length=max_snippet_length, cross_fade=cross_fade)


def run_dir_backend(input_dir, output_path, feature_set, output_length = 10, max_snippet_length = 0.5, cross_fade = 50, extension=".mp3"):
    '''
    Runs a concatenation job on a directory. The directory provided and its subdirectories are recursively
    searched and any files matching the provided extension (default mp3) will be conctatenated into a single
    output file. 
    
    :param input_dir: The directory root to use as a basis to recursively load audio files from
    :param output_path: Path to output the concatenated audio to
    :param feature_set list of feature names (e.g. 'rms', 'pitch') to be used in the audio analysis
    :param output_length: Desired final output length in seconds
    :param max_snippet_length: The maximum length of each sample when concatenating (seconds)
    :param cross_fade: Cross fade length between samples (milliseconds)
    :param extension: The file audio file extension type to search for
    '''

    if max_snippet_length * 1000 <= cross_fade:
        logger.error(f"Snippet length ({max_snippet_length} s) must be bigger than cross fade ({cross_fade} ms)")
        sys.exit(1)

    audio_dir = Path(input_dir)
    files = list(audio_dir.rglob(f"*{extension}"))
    run_concatenator(files, output_path=output_path, output_length=output_length, feature_set=feature_set, max_snippet_length=max_snippet_length, cross_fade=cross_fade)


def main():
    '''
    This is just a basic example to demonstrate how to use the concatenative synth
    to load files directly from disk and create a single stitched output file.
    '''
    run_dir_backend(download_directory, current_directory)


if __name__=="__main__":
    main()
    
