from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any
from pathlib import Path
import numpy as np
import freesound
import random
import os
import librosa
from yt_dlp import YoutubeDL

class AudioSnippet:
    def __init__(
            self,
            samples: np.ndarray,
            sample_rate: int,
            metadata: Dict[str, Any] | None = None,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.metadata = metadata or {}

class AudioBackend(ABC):
    @abstractmethod
    def get_snippets(self) -> Iterable[AudioSnippet]:
        pass

class FreesoundBackend(AudioBackend):
    API_KEY = os.environ.get("FREESOUND_API_KEY")

    def __init__(self, 
                 search_terms, 
                 output_path, 
                 target_sr = 44100,
                 min_per_term = 5, 
                 max_per_term = 10, 
                 duration_range=(0.1, 0.5)):
        self.client = freesound.FreesoundClient()
        self.search_terms = search_terms
        self.output_path = output_path
        self.target_sr = target_sr
        self.min_per_term = min_per_term
        self.max_per_term = max_per_term
        self.duration_range = duration_range

        self.client.set_token(self.API_KEY, "token")

    
    def _download_preview(self, sound, out_dir):
        '''
        Downloads the "previews" for each sound provided
        These are downloaded in hq & in mp3 format
        
        :param sounds: List of Sound instances
        :param out_dir: Directory to save the output files in
        '''
        sound_name= Path(sound.name).stem
        filename = sound_name + ".mp3"
        sound.retrieve_preview(out_dir, filename, quality="hq")

        return out_dir / filename

    
    def get_snippets(self):
            filter_str = (
                f"duration:[{self.duration_range[0]} TO {self.duration_range[1]}]"
            )

            for term in self.search_terms:
                num_results = random.randint(self.min_per_term, self.max_per_term)
                results = self.client.search(
                    query = term,
                    fields="id,name,previews",
                    filter = filter_str,
                    page_size=num_results
                )

                for sound in results:
                    sound_path = self._download_preview(sound, self.output_path)
                    samples, sr = librosa.load(sound_path, sr=self.target_sr)

                    yield AudioSnippet(
                        samples=samples,
                        sample_rate=sr,
                        metadata={
                            "source": "freesound",
                            "id": sound.id,
                            "name": sound.name,
                        }
                    )
                
class SliceDuration:
    def __init__(self, length = 0.5):
        self.length = length

    def __call__(self, info, ydl):
        duration = info.get("duration")
        if not duration:
            return

        middle = duration / 2
        start = max(0.0, middle - self.length/2)
        end = min(duration, middle + self.length/2)
        yield {"start_time": start, "end_time": end}


class DownloadTracker:
    def __init__(self):
        self.files = []

    def __call__(self, info):
        if info.get("status") == "finished":
            self.files.append(info["filename"])


class YoutubeBackend(AudioBackend):
    def __init__(self, 
                 search_terms,
                 output_path,
                 target_sr = 44100,
                 batch_size = 32,
                 words_per_phrase = 2):
        self.search_terms = search_terms
        self.ouput_path = output_path
        self.target_sr = target_sr
        self.batch_size = batch_size
        self.words_per_phrase = words_per_phrase 

    def _get_random_phrase(self):
        words = random.sample(self.search_terms, self.words_per_phrase)
        return " ".join(words)

    def _ydl_opts(self, download_dir, tracker):

        return {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "paths": {"home": str(download_dir)},
            "outtmpl": "%(id)s.%(ext)s",
            "download_ranges": SliceDuration(random.uniform(0.1, 1.0)),
            "progress_hooks": [tracker],
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "postprocessor_args": ["-loglevel", "error"],
        }
    
    def get_snippets(self):
        for _ in range(self.batch_size):
            phrase = self._get_random_phrase()
            query = f'ytsearch1:"{phrase}"'

            tracker = DownloadTracker()

            with YoutubeDL(self._ydl_opts(self.ouput_path, tracker)) as ydl:
                ydl.download([query])

            for filename in tracker.files:
                audio_path = Path(filename).with_suffix(".mp3")
                samples, sr = librosa.load(audio_path, sr = 44100)

            yield AudioSnippet(
                samples=samples,
                sample_rate=44100,
                metadata={
                    "source": "youtube",
                    "query": phrase
                }
            )