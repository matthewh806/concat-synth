import random
import os
import sys
import argparse
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


def run_download_backend(backend_name, words_path, output_path, max_snippets):
    words = load_words(words_path)

    if backend_name == "youtube":
        backend = YoutubeBackend(download_directory)
        queries = [get_random_phrase(words) for _ in range(max_snippets)]
    else: # freesound
        backend = FreesoundBackend(download_directory)
        queries = words

    snippets = collect_snippets_parallel(backend, queries, max_snippets)

    if len(snippets) == 0:
        print(f"No audio files found in download directory!")
        sys.exit(1)

    concatenated = concatenate_snippets(snippets)
    sf.write(output_path, concatenated, 44100)


def run_dir_backend(input_dir, output_path):
    pass


def main():
    parser = argparse.ArgumentParser("Concatenative Audio Synthesis")

    subparsers = parser.add_subparsers(dest="command")

    #---------------------------------
    # Subcommand: Download and concat
    #---------------------------------
    download_parser = subparsers.add_parser("download", help="Download audio from backend and concatenate")
    download_parser.add_argument(
        "backend", choices=["youtube", "freesound"],
        help="Backend to use for downloading audio"
    )
    download_parser.add_argument(
        "--words", type=str, default="words.txt",
        help="Path to the word list for phrase generation"
    )
    download_parser.add_argument(
        "--out", type=str, default="output.wav",
        help="Output WAV file path"
    )
    download_parser.add_argument(
        "--max-snippets", type=int, default=32,
        help="Max number of snippets to download"
    )

    #---------------------------------
    # Subcommand: Use local files
    #---------------------------------
    dir_parser = subparsers.add_parser("dir", help="Use existing audio files from directory")
    dir_parser.add_argument("input_dir, type=str", help="Directory containing audio files")
    dir_parser.add_argument(
        "--out", type=str, default="output.wav",
        help="Output WAV file path"
    )

    args = parser.parse_args()
    if args.command == "download":
        run_download_backend(
            backend_name = args.backend,
            words_path = args.words,
            output_path = args.out,
            max_snippets = args.max_snippets
        )
    elif args.command == "dir":
        run_dir_backend(
            input_dir= args.input_dir,
            output_path= args.out
        )
    else:
        parser.print_help()
    

if __name__=="__main__":
    main()
    
