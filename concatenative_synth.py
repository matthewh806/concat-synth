import freesound
import random
import os
import sys
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

API_KEY = os.environ.get("FREESOUND_API_KEY")

current_directory = Path()
download_directory = current_directory / "audio_downloads"

def generate_random_phrases(words, phrase_length=2, num_phrases=5):
    '''
    Given a list of words will randomly generate a list of phrases 
    of a specified length

    This doesn't really work well in the case of the freesound API...
    So its currently unused
    
    :param words: List of words to be used in phrase creation
    :param phrase_length: Number of words in each phrase
    :param num_phrases: Number of phrases to generate

    :return list of generated phrases
    '''
    phrases = []
    for _ in range(num_phrases):
        phrase = " ".join(random.sample(words, phrase_length))
        phrases.append(phrase)
    
    return phrases


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


def search_sounds_for_word(client, 
                           word, 
                           min_results=5,
                           max_results=10,
                           duration_range=(0.1, 0.5)):
    '''
    Searches for sounds matching a word or phrase
    The actual number of results returned will be 
    a randomised value between min_results & max_results
    OR fewer if less results than min_results are found
    
    :param client: Freesound client instance
    :param word: Search term
    :param min_results: Minimum number of results to return
    :param max_results: Maximum number of results to return
    :param duration_range: Duration range for a sound (seconds)
    '''
    
    num_results = random.randint(min_results, max_results)
    
    filter_str = (
        f"duration:[{duration_range[0]} TO {duration_range[1]}]"
    )

    results = client.search(
        query = word,
        fields="id,name,previews",
        filter = filter_str,
        page_size=num_results
    )

    return list(results)


def download_previews(sounds, out_dir):
    '''
    Downloads the "previews" for each sound provided
    These are downloaded in hq & in mp3 format
    
    :param sounds: List of Sound instances
    :param out_dir: Directory to save the output files in
    '''
    for sound in sounds:
        sound_name= Path(sound.name).stem
        filename = sound_name + ".mp3"
        sound.retrieve_preview(out_dir, filename, quality="hq")
    

def fetch_corpus(client, words):
    '''
    Creates a corpus of words and the associated sounds instances
    For each word in the corpus there is a list of sound instances
    
    :param client: Freesound client instance
    :param words: List of words to use to generate the corpus

    :return dictionary containing sound instances for each word
    '''
    corpus = {}

    for word in words:
        corpus[word] = search_sounds_for_word(client, word)
        print(f"Found {len(corpus[word])} sounds for word: {word}")

    return corpus


def load_snippets(directory, target_sr = 44100):
    '''
    Loads the audio snippets into numpy arrays
    Does sample rate conversion if neccessary to ensure
    they are all consistent
    
    :param directory: location of the audio files
    :param target_sr: sample rate to convert to

    :return list of np.array representations of the sound
    '''
    snippets = []

    for path in directory.iterdir():
        if path.is_file() and path.suffix == ".mp3":
            snippet, _ = librosa.load(path, sr=target_sr)
            snippets.append(snippet)

    return snippets


def concatenate_snippets(snippets, output_sr = 44100):
    '''
    Generates a single concatenated output file randomly
    from the snippets provided
    
    :param snippets: List of sample data as numpy arrays
    :param output_sr: Sample rate to save the output as

    :return concatenated audio as a numpy array
    '''
    total_num_samples = sum( len(snippet) for snippet in snippets)
    print(f"Generating a concatenated file of length {total_num_samples / output_sr}")
    
    random.shuffle(snippets)
    output = np.concatenate(snippets)
    return output


if __name__=="__main__":
    print("Concatenative Synth")

    words = load_words("words.txt")

    client = freesound.FreesoundClient()
    client.set_token(API_KEY, "token")

    corpus = fetch_corpus(client, words)

    for word, sounds in corpus.items():
        print(f"Downloading sounds for {word}")
        download_previews(sounds, download_directory)

    snippets = load_snippets(download_directory)
    if len(snippets) == 0:
        print(f"No audio files found in download directory!")
        sys.exit(1)

    concatenated = concatenate_snippets(snippets)
    sf.write('output.wav', concatenated, 44100)
    
