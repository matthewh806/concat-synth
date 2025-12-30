from .audio_downloader import AudioDownloader
from pathlib import Path
import random
import uuid
from yt_dlp import YoutubeDL


class SliceDuration:
    '''
    SliceDuration callable method for determining the
    the size clip to extract by yt-dlp
    '''
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
    '''
    Callable used by yt-dlp to get information on
    download status
    '''
    def __init__(self):
        self.files = []

    def __call__(self, info):
        if info.get("status") == "finished":
            self.files.append(info["filename"])


class YoutubeAudioDownloader(AudioDownloader):
    '''
    AudioDownloader backend implementation which uses
    the yt-dlp to download audio samples
    https://github.com/yt-dlp/yt-dlp
    '''
    
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
        '''
        :param query: string to use as the query when calling the API

        :return paths of the downloaded files in a list
        '''
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