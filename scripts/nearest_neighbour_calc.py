from concatenative.core import analyse_snippets, nearest_neighbour_search
from concatenative.core import audio_loader
from concatenative.core.logger import setup_logger
from pathlib import Path
import sys
import random
import logging

ROOT = Path(__file__).resolve().parents[1]

'''
This script loads all of the files in the audio_downloads directory, analyses each AudioSnippet
and performs a nearest neighbour search using a random AudioSnippet from the list
'''

if __name__ == "__main__":

    setup_logger(log_level=logging.DEBUG)

    audio_dir = Path(ROOT / "audio_downloads")
    file_paths = list(audio_dir.rglob(f"*{'.mp3'}"))

    if len(file_paths) == 0:
        logging.error(f"No audio files found!")
        sys.exit(1)

    snippets = [snip for path in file_paths if (snip := audio_loader(path, max_clip_length=0.2)) is not None]
    target = snippets[random.randint(0, len(snippets)-1)]
    analyse_snippets(snippets)

    logging.info(f"Finding nearest neighbour for target: {target}")
    
    nearest_neighbour = nearest_neighbour_search(
        snippets=snippets,
        target_snippet=target
    )

    if nearest_neighbour:
        logging.info(f"Found Nearest neighbour: {nearest_neighbour}")