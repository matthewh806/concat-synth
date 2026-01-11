from concatenative.audio import audio_loader
from concatenative.analysis.corpus import Corpus
from concatenative.analysis.features import Feature
from concatenative.analysis.feature_extractor import FeatureExtractor
from concatenative.utils.logger import setup_logger
from pathlib import Path
import sys
import logging

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parents[1]

'''
This script loads all of the files in the audio_downloads directory, analyses each AudioSnippet
and performs a nearest neighbour search using a random AudioSnippet from the list
'''

features = [
    Feature(
        name='rms',
        extractor = lambda samples, _ : np.mean(librosa.feature.rms(y = samples))
    ),
    Feature(
        name='pitch',
        extractor = lambda samples, sr : np.mean(librosa.pyin(y=samples, fmin=50, fmax=5000, sr=sr))
    ),
    Feature (
        name="spectral centroid",
        extractor = lambda samples, sr : np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr))
    )
]

if __name__ == "__main__":

    setup_logger(log_level=logging.DEBUG)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = list(audio_dir.rglob(f"*{'.mp3'}"))

    if len(file_paths) == 0:
        logging.error(f"No audio files found!")
        sys.exit(1)

    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=0.2)) is not None]
    corpus = Corpus(snippets, FeatureExtractor(features=features))
    target = corpus.get_random_snippet()

    logging.info(f"Finding nearest neighbour for target: {target}")
    
    nearest_neighbour = corpus.nearest_neighbour_search(
        target_snippet=target,
        exclusion_list=[]
    )

    if nearest_neighbour:
        logging.info(f"Found Nearest neighbour: {nearest_neighbour}")