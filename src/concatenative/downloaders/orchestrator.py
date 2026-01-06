from concatenative.utils import timed
from concatenative.utils import run_parallel_io_tasks
import logging

logger = logging.getLogger(__name__)

@timed
def collect_snippets_parallel(
        backend,
        queries,
):
    '''
    Calls a parallelised task manager for downloading / retrieving audio. 

    :param backend: The source of the audio files to be downloaded (e.g. FreesoundAudioDownloader, YoutubeAudioDownloader)
    :param queries: List of the query strings to download

    :return List of paths to the audio files which have been downloaded
    '''
    
    snippets = []

    def task_complete_callback(result):
        '''
        :param result: result will contain a list of paths to audio files downloaded by the backend
        '''
        for snippet in result:
            snippets.append(snippet)

    run_parallel_io_tasks(backend.get_snippets, queries, task_complete_callback=task_complete_callback)
    return snippets
