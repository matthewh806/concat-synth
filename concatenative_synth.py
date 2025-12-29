import random
import os
import sys
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from backends import FreesoundAudioDownloader, YoutubeAudioDownloader
from orchestrator import collect_snippets_parallel
from audio_loader import audio_loader

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


def concatenate_snippets(snippets, output_sr = 44100, cross_fade = 50):
    '''
    Generates a single concatenated output file randomly
    from the snippets provided
    
    :param snippets: List of sample data as numpy arrays
    :param output_sr: Sample rate to save the output as
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


def run_download_backend(backend_name, words_path, output_path, max_snippets = 64, max_snippet_length = 0.5, cross_fade = 50):
    words = load_words(words_path)

    if backend_name == "youtube":
        backend = YoutubeAudioDownloader(download_directory)
        queries = [get_random_phrase(words) for _ in range(max_snippets)]
    else: # freesound
        queries = words
        results_per_word = 1 if max_snippets <= len(words) else int(len(words) / max_snippets)
        backend = FreesoundAudioDownloader(download_directory, number_of_results=results_per_word)

    download_paths = collect_snippets_parallel(backend, queries, max_snippets)

    if len(download_paths) == 0:
        print(f"No audio files found in download directory!")
        sys.exit(1)

    snippets = [snip for path in download_paths if (snip := audio_loader(path, max_clip_length=max_snippet_length)) is not None]
    concatenated = concatenate_snippets(snippets, cross_fade=cross_fade)
    sf.write(output_path, concatenated, 44100)


def run_dir_backend(input_dir, output_path, max_snippet_length = 0.5, cross_fade = 50, extension=".mp3"):
    audio_dir = Path(input_dir)
    files = list(audio_dir.rglob(f"*{extension}"))

    if len(files) == 0:
        print(f"No audio files found in {input_dir}!")
        sys.exit(1)

    snippets = [snip for path in files if (snip := audio_loader(path, max_clip_length=max_snippet_length)) is not None]
    concatenated = concatenate_snippets(snippets, cross_fade=cross_fade)
    sf.write(output_path, concatenated, 44100)


def main():

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--out", type=str, default="output.wav",
        help="Output WAV file path"
    )
    parent_parser.add_argument(
        "--max-slice-length", type=float, default=0.5,
        help="Maximum length of each slice (seconds)"
    )
    parent_parser.add_argument(
        "--fade", type=int, default=50,
        help="Cross fade length (milliseconds)"
    )

    parser = argparse.ArgumentParser("Concatenative Audio Synthesis")
    subparsers = parser.add_subparsers(dest="command")

    #---------------------------------
    # Subcommand: Download and concat
    #---------------------------------
    download_parser = subparsers.add_parser("download", parents=[parent_parser], help="Download audio from backend and concatenate")
    download_parser.add_argument(
        "backend", choices=["youtube", "freesound"],
        help="Backend to use for downloading audio"
    )
    download_parser.add_argument(
        "--words", type=str, default="words.txt",
        help="Path to the word list for phrase generation"
    )

    download_parser.add_argument(
        "--max-snippets", type=int, default=32,
        help="Max number of snippets to download"
    )

    #---------------------------------
    # Subcommand: Use local files
    #---------------------------------
    dir_parser = subparsers.add_parser("dir", parents=[parent_parser], help="Use existing audio files from directory")
    dir_parser.add_argument("input_dir", type=str, help="Directory containing audio files")

    args = parser.parse_args()
    if args.command == "download":
        run_download_backend(
            backend_name = args.backend,
            words_path = args.words,
            output_path = args.out,
            max_snippets = args.max_snippets,
            max_snippet_length=args.max_slice_length,
            cross_fade=args.fade
        )
    elif args.command == "dir":
        run_dir_backend(
            input_dir= args.input_dir,
            output_path= args.out,
            max_snippet_length=args.max_slice_length,
            cross_fade=args.fade
        )
    else:
        parser.print_help()
    

if __name__=="__main__":
    main()
    
