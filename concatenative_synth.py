import freesound
import random
import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

API_KEY = os.environ.get("FREESOUND_API_KEY")

current_directory = Path()
download_directory = current_directory / "audio_downloads"

def search_sounds_for_word(client, 
                           word, 
                           min_results=5,
                           max_results=10,
                           duration_range=(0.1, 0.5)):
    
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
    for sound in sounds:
        sound_name= Path(sound.name).stem
        filename = sound_name + ".mp3"
        sound.retrieve_preview(download_directory, filename, quality="hq")
    

def fetch_corpus(client, words):
    corpus = {}
    for word in words:
        corpus[word] = search_sounds_for_word(client, word)
    return corpus


def load_snippets(directory, target_sr = 44100):
    snippets = []

    for path in directory.iterdir():
        if path.is_file() and path.suffix == ".mp3":
            snippet, _ = librosa.load(path, sr=target_sr)
            snippets.append(snippet)

    return snippets


def concatenate_snippets(snippets, output_sr = 44100):
    total_num_samples = sum( len(snippet) for snippet in snippets)
    #concatenated = np.zeros(total_num_samples)

    print(f"Generating a concatenated file of length {total_num_samples / output_sr}")
    random.shuffle(snippets)

    output = np.concatenate(snippets)
    return output


if __name__=="__main__":
    print("Concatenative Synth")

    words = ["Elephant", "Sunshine", "Harmony", "Labyrinth", "Melody", "Ethereal"]

    client = freesound.FreesoundClient()
    client.set_token(API_KEY, "token")

    corpus = fetch_corpus(client, words)

    for word, sounds in corpus.items():
        print(f"Downloading sounds for {word}")
        download_previews(sounds, download_directory)

    snippets = load_snippets(download_directory)
    concatenated = concatenate_snippets(snippets)
    sf.write('output.wav', concatenated, 44100)
    
