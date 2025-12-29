import random
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from backends import AudioSnippet, FreesoundBackend, YoutubeBackend
from orchestrator import collect_snippets_parallel

API_KEY = os.environ.get("FREESOUND_API_KEY")

current_directory = Path()
download_directory = current_directory / "audio_downloads"

def get_random_phrase(word_list, phrase_len = 2):
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
    return words[:limit + 1] if len(words) >= limit else words


def concatenate_snippets(snippets, output_sr = 44100):
    '''
    Generates a single concatenated output file randomly
    from the snippets provided
    
    :param snippets: List of sample data as numpy arrays
    :param output_sr: Sample rate to save the output as

    :return concatenated audio as a numpy array
    '''
    total_num_samples = sum( len(snippet.samples) for snippet in snippets)
    print(f"Generating a concatenated file of length {total_num_samples / output_sr}")
    
    random.shuffle(snippets)
    output = np.concatenate([snippet.samples for snippet in snippets])
    return output


if __name__=="__main__":
    print("Concatenative Synth")

    words = load_words("words.txt")
    backend = YoutubeBackend(download_directory)
    max_snippets = 256
    queries = [get_random_phrase(words) for _ in range(max_snippets)]
    snippets = collect_snippets_parallel(backend, queries, max_snippets)

    if len(snippets) == 0:
        print(f"No audio files found in download directory!")
        sys.exit(1)

    concatenated = concatenate_snippets(snippets)
    sf.write('output.wav', concatenated, 44100)
    
