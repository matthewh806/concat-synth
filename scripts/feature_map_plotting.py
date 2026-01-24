from pathlib import Path
import matplotlib.pyplot as plt
import sounddevice as sd
from concatenative.audio.audio_loader import audio_loader, find_audio_files_recursively
from concatenative.analysis.corpus import Corpus
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.path.selector import generate_concatenation_path
from concatenative.visualisation.plotting import InteractiveCorpusPlot
from concatenative.utils.logger import setup_logger
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
    logger = logging.getLogger(__name__)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = find_audio_files_recursively(audio_dir)

    if len(file_paths) == 0:
        logger.error(f"No audio files found!")
        sys.exit(1)

    rms = FEATURE_REGISTRY['rms']
    spectral_centroid = FEATURE_REGISTRY['spectral centroid']
    pitch = FEATURE_REGISTRY['pitch']
    feature_set = [
        rms, spectral_centroid, pitch
    ]

    snippets = [
        snippet
        for file_path in file_paths
        for snippet in audio_loader(
            file_path, max_clip_length = 0.2, segmentation_stratgy='slices', segment_duration_s=0.1, max_snippets=1
        )
    ]
    corpus = Corpus(snippets, FeatureExtractor(features=feature_set))
    print(f"Number of duplicate snippets: {corpus.get_number_of_duplicates()}")
    concatenation_path = generate_concatenation_path(corpus=corpus, output_length_sec=20)

    print(concatenation_path.get_stats())    

    plot = InteractiveCorpusPlot(corpus.snippets, 
                                 rms, 
                                 spectral_centroid, 
                                 pitch, 
                                 normalised=False,
                                 on_click_callback=play_snippet_audio_callback, 
                                 path_to_draw=concatenation_path)
    plt.show()