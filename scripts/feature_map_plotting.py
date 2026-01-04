from pathlib import Path
import matplotlib.pyplot as plt
import sounddevice as sd
from concatenative.core import audio_loader, AudioSnippet
from concatenative.core.corpus import Corpus
from concatenative.core.selector import generate_concatenation_path
from concatenative.utils.plotting import InteractiveCorpusPlot
from concatenative.core.logger import setup_logger
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]

def play_snippet_audio_callback(snippet):
    '''
    Callback method which is passed to InteractiveCorpusPlot
    Will be called when a point in the scatter plot is clicked
    And uses sounddevice to playback the snippets audio
    
    :param snippet: AudioSnippet to playback
    '''

    if not snippet:
        return
    
    try:
        sd.stop() 
        sd.play(snippet.samples, snippet.sample_rate, blocking=False)
    except Exception as e:
        print(f"Error playing audio: {e}")


if __name__ == "__main__":

    setup_logger(log_level=logging.DEBUG)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = list(audio_dir.rglob(f"*{'.mp3'}"))

    if len(file_paths) == 0:
        logging.error(f"No audio files found!")
        sys.exit(1)

    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=0.2)) is not None]
    corpus = Corpus(snippets)
    concatenation_path = generate_concatenation_path(corpus=corpus, output_length_sec=5)

    plot = InteractiveCorpusPlot(corpus.snippets, 'rms', 'spectral_centroid', 'pitch', on_click_callback=play_snippet_audio_callback, path_to_draw=concatenation_path)
    plt.show()