import random
import os
import sys
import math
import numpy as np
import soundfile as sf
from pathlib import Path
from .downloaders import FreesoundAudioDownloader, YoutubeAudioDownloader
from .core import collect_snippets_parallel, audio_loader

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


def normalise_features(snippets):
    '''
    Normalises features to be in the range [0,1]

    TODO: MOVE
    '''

    feature_bounds = {
        'rms': {'min': float('inf'), 'max': float('-inf')},
        'pitch': {'min': float('inf'), 'max': float('-inf')},
        'spectral_centroid': {'min': float('inf'), 'max': float('-inf')}
    }

    for snippet in snippets:
        for feature_name, value in snippet.features.items():
            print(f"{feature_name}: {value:.4f}")
            if value < feature_bounds[feature_name]['min']:
                feature_bounds[feature_name]['min'] = value
            
            if value > feature_bounds[feature_name]['max']:
                feature_bounds[feature_name]['max'] = value
    
    print(feature_bounds)

    for snippet in snippets:
        snippet.normalised_features = {}
        for feature_name, value in snippet.features.items():
            feat_min = feature_bounds[feature_name]['min']
            feat_max = feature_bounds[feature_name]['max']

            if np.isnan(value):
                snippet.normalised_features[feature_name] = value
            else:
                normalised_value = (value - feat_min) / (feat_max - feat_min) 
                snippet.normalised_features[feature_name] = normalised_value

            print(f"{feature_name}: {value:.4f}, normalised: {snippet.normalised_features[feature_name]:.4f}")
        
            


def concatenate_snippets(snippets, output_sr = 44100, cross_fade = 50):
    '''
    Generates a single concatenated output file randomly
    from the snippets provided
    
    :param snippets: List of sample data as numpy arrays
    :param output_sr: Sample rate to save the output as (Hz)
    :param cross_fade: Cross fade length (milliseconds)

    :return concatenated audio as a numpy array
    '''
    random.shuffle(snippets)
    output = snippets[0].samples.copy()
    cross_fade_samples = int((cross_fade / 1000) * output_sr)

    for snippet in snippets[1:]:
        samples = snippet.samples
        cross_fade_amount = len(samples) if len(samples) < cross_fade_samples else cross_fade_samples
        fade_out = output[-cross_fade_amount:] * np.linspace(1, 0, cross_fade_amount)
        fade_in = samples[:cross_fade_amount] * np.linspace(0, 1, cross_fade_amount)
        overlapping_region = fade_out + fade_in
        output = np.concatenate([output[:-cross_fade_amount], overlapping_region, samples[cross_fade_amount:]])

    print(f"Generated a concatenated file of length {(len(output) / output_sr):.2f} seconds")
    return output


def run_concatenator(file_paths, output_path, max_snippet_length = 0.5, cross_fade = 50):
    '''
    Loads the audio files, concatenates them and outputs the audio 
    TODO: separate this out better and rename method
    '''
    if len(file_paths) == 0:
        print(f"No audio files found!")
        sys.exit(1)

    print(f"Loading {len(file_paths)} files into the concatenator")
    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=max_snippet_length)) is not None]
    normalise_features(snippets)
    concatenated = concatenate_snippets(snippets, cross_fade=cross_fade)
    sf.write(output_path, concatenated, 44100)


def run_download_backend(backend_name, words_path, output_path, max_snippets = 64, max_snippet_length = 0.5, cross_fade = 50):
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
    :param max_snippets: The maximum number of audio samples to download
    :param max_snippet_length: The maximum length of each sample when concatenating (seconds)
    :param cross_fade: Cross fade length between samples (milliseconds)
    '''

    print(f"Running download backend: {backend_name}, retrieving: {max_snippets} files")

    words = load_words(words_path)

    if backend_name == "youtube":
        backend = YoutubeAudioDownloader(download_directory)
        queries = [get_random_phrase(words) for _ in range(max_snippets)]
    else: # freesound
        queries = words
        results_per_word = 1 if max_snippets <= len(words) else math.ceil(len(words) / max_snippets)
        backend = FreesoundAudioDownloader(download_directory, number_of_results=results_per_word)

    download_paths = collect_snippets_parallel(backend, queries, max_snippets)
    run_concatenator(download_paths, output_path=output_path, max_snippet_length=max_snippet_length, cross_fade=cross_fade)


def run_dir_backend(input_dir, output_path, max_snippet_length = 0.5, cross_fade = 50, extension=".mp3"):
    '''
    Runs a concatenation job on a directory. The directory provided and its subdirectories are recursively
    searched and any files matching the provided extension (default mp3) will be conctatenated into a single
    output file. 
    
    :param input_dir: The directory root to use as a basis to recursively load audio files from
    :param output_path: Path to output the concatenated audio to
    :param max_snippet_length: The maximum length of each sample when concatenating (seconds)
    :param cross_fade: Cross fade length between samples (milliseconds)
    :param extension: The file audio file extension type to search for
    '''
    audio_dir = Path(input_dir)
    files = list(audio_dir.rglob(f"*{extension}"))
    run_concatenator(files, output_path=output_path, max_snippet_length=max_snippet_length, cross_fade=cross_fade)


def main():
    '''
    This is just a basic example to demonstrate how to use the concatenative synth
    to load files directly from disk and create a single stitched output file.
    '''
    run_dir_backend(download_directory, current_directory)


if __name__=="__main__":
    main()
    
