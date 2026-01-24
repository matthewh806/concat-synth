from pathlib import Path
from concatenative.utils.logger import setup_logger
from concatenative.analysis.segmentation import segment_audio
from concatenative.visualisation.plotting import plot_signal_segmentation
import logging
import librosa

ROOT = Path(__file__).resolve().parents[1]

SEGMENTATION = 'slices'

if __name__ == "__main__":
    setup_logger(log_level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    audio_dir = Path(ROOT / "audio_downloads")
    audio_path = audio_dir / "Aurorae Aurorae__d24c7a/yX8bYwl9rKc.wav"

    samples, sr = librosa.load(audio_path, sr=44100)
    segments = segment_audio(samples, sr, strategy=SEGMENTATION, segment_duration_s=0.05)

    print(f"Num segments: {len(segments)}")
    plot_signal_segmentation(samples=samples, segments=segments)
