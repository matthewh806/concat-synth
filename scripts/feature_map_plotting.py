from pathlib import Path
from concatenative.core import audio_loader
from concatenative.core.corpus import Corpus
from concatenative.utils.plotting import plot_feature_map
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = list(audio_dir.rglob(f"*{'.mp3'}"))

    if len(file_paths) == 0:
        logging.error(f"No audio files found!")
        sys.exit(1)

    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=0.2)) is not None]
    corpus = Corpus(snippets)

    plot_feature_map(corpus.snippets, 'rms', 'spectral_centroid', 'pitch')