from pathlib import Path
from concatenative import run_download_backend
from concatenative.utils.logger import setup_logger
import logging

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    setup_logger(log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    run_download_backend("youtube", ROOT / "data/words.txt", ROOT / "output.wav", output_length=60, max_snippets=256, max_snippet_length=0.3)