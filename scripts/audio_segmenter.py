from pathlib import Path
from concatenative.utils.logger import setup_logger
from concatenative.analysis.segmentation import segment_audio
from concatenative.visualisation.plotting import plot_signal_segmentation
from concatenative.config import load_config
import logging
import librosa

ROOT = Path(__file__).resolve().parents[1]

SEGMENTATION = 'slices'

if __name__ == "__main__":
    setup_logger(log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    audio_dir = Path(ROOT / "audio_downloads")
    audio_path = audio_dir / "Aurora Halo__23b82a/e4ObYW5aH74.wav"

    samples, sr = librosa.load(audio_path, sr=44100)
    segments = segment_audio(samples, sr, strategy=SEGMENTATION, config=load_config())

    print(f"Num segments: {len(segments)}")
    plot_signal_segmentation(samples=samples, segments=segments)
