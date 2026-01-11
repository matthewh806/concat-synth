from .audio_downloader import AudioDownloader
from pathlib import Path
import threading
import random
import uuid
from yt_dlp import YoutubeDL
import logging

logger = logging.getLogger(__name__)

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

        self.seen_video_ids = set()

        # This lock prevents race conditions when accessing self.seen_video_ids
        self.lock = threading.Lock()

    def _ydl_download_opts(self, download_dir, tracker, slice_dur = 0.1):

        return {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "ignore_errors": True,
            "paths": {"home": str(download_dir)},
            "outtmpl": "%(id)s.%(ext)s",
            "download_ranges": SliceDuration(slice_dur),
            "progress_hooks": [tracker],
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "postprocessor_args": ["-loglevel", "error"],
        }
    
    def download_audio(self, query, num_search_results = 5):
        '''
        Thread safe audio downloader for youtube that uses the yt-dlp library
        Prevents duplicate downloads within a single run.

        It will query num_search_results and go through the results
        until it finds a result which hasn't already been downloaded

        :param query: string to use as the query when calling the API
        :num_search_results: Number of search results to query

        :return paths of the downloaded files in a list
        '''
        
        # 1. Fetch metadata for top N search results of query
        logging.info(f"yt-dlp getting metadata for query: {query}")
        ydl_metadata_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist'
        }

        try:
            with YoutubeDL(ydl_metadata_opts) as ydl:
                search_result = ydl.extract_info(f"ytsearch{num_search_results}:{query}", download=False)
                entries = search_result.get('entries', [])
        except Exception as e:
            logger.error(f"Failed fetching metadata for '{query}': {e}")
            return []

        video_to_download_id = None 
        with self.lock:
            for entry in entries:
                video_id = entry.get('id')
                if video_id and video_id not in self.seen_video_ids:
                    self.seen_video_ids.add(video_id)
                    video_to_download_id = video_id
        
        if not video_to_download_id:
            logger.warning(f"yt-dlp no more videos to download for {query}")
            return []

        logging.info(f"yt-dlp download starting for query: {query}")
        tracker = DownloadTracker()

        # TODO: pass in as param
        slice_dur = random.uniform(0.1, 0.5)

        run_id = uuid.uuid4().hex[:6]
        query_dir = self.ouput_path / f"{query}__{run_id}"
        query_dir.mkdir(parents=True, exist_ok = True)
        with YoutubeDL(self._ydl_download_opts(query_dir, tracker, slice_dur)) as ydl:
            try:
                ydl.download([video_to_download_id])
            except Exception as e:
                logging.error(f"yt-dlp failed for query '{query}': {e}")
                return []

        paths = []
        for filename in tracker.files:
            audio_path = Path(filename).with_suffix(".mp3")
            paths.append(audio_path)

        return paths