from abc import ABC, abstractmethod
from pathlib import Path
import freesound
import random
import os
import uuid
from yt_dlp import YoutubeDL


class AudioDownloader(ABC):
    @abstractmethod
    def get_snippets(self, query) -> list[Path]:
        pass


class FreesoundAudioDownloader(AudioDownloader):
    API_KEY = os.environ.get("FREESOUND_API_KEY")

    def __init__(self,  
                 output_path, 
                 target_sr = 44100, 
                 number_of_results = 10, 
                 duration_range=(0.1, 0.5)):
        self.client = freesound.FreesoundClient()
        self.output_path = output_path
        self.target_sr = target_sr
        self.number_of_results = number_of_results
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

    def get_snippets(self, query):
            filter_str = (
                f"duration:[{self.duration_range[0]} TO {self.duration_range[1]}]"
            )

            results = self.client.search(
                query = query,
                fields="id,name,previews",
                filter = filter_str,
                page_size=self.number_of_results
            )

            paths = []
            for sound in results:
                sound_path = self._download_preview(sound, self.output_path)
                paths.append(sound_path)
        
            return paths
                

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

        # avoid tiny ranges
        if end - start < 0.05:
            return

        yield {"start_time": start, "end_time": end}


class DownloadTracker:
    def __init__(self):
        self.files = []

    def __call__(self, info):
        if info.get("status") == "finished":
            self.files.append(info["filename"])


class YoutubeAudioDownloader(AudioDownloader):
    def __init__(self, 
                 output_path,
                 target_sr = 44100):
        self.ouput_path = output_path
        self.target_sr = target_sr

    def _ydl_opts(self, download_dir, tracker, slice_dur = 0.1):

        return {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "ignore_errors": True,
            "paths": {"home": str(download_dir)},
            "outtmpl": "%(id)s.%(ext)s",
            "download_ranges": SliceDuration(0.1),
            "progress_hooks": [tracker],
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "postprocessor_args": ["-loglevel", "error"],
        }
    
    def get_snippets(self, query):
        print(f"yt-dlp download starting for query: {query}")

        search_query = f'ytsearch1:"{query}"'
        tracker = DownloadTracker()

        # TODO: pass in as param
        slice_dur = random.uniform(0.1, 0.5)

        run_id = uuid.uuid4().hex[:6]
        query_dir = self.ouput_path / f"{query}__{run_id}"
        query_dir.mkdir(parents=True, exist_ok = True)
        with YoutubeDL(self._ydl_opts(query_dir, tracker, slice_dur)) as ydl:
            try:
                ydl.download([search_query])
            except Exception as e:
                print(f"yt-dlp failed for query '{query}': {e}")
                return []

        paths = []
        for filename in tracker.files:
            audio_path = Path(filename).with_suffix(".mp3")
            paths.append(audio_path)

        return paths