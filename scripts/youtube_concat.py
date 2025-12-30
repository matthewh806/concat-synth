from pathlib import Path
from concatenative import run_download_backend

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    run_download_backend("youtube", ROOT / "data/words.txt", ROOT / "output.wav")