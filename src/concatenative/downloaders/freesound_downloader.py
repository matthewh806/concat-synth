from .audio_downloader import AudioDownloader
from pathlib import Path
import freesound
import os

def get_freesound_api_key():
     class MissingAPIKeyError(RuntimeError):
        pass

     key = os.environ.get("FREESOUND_API_KEY")
     if not key:
          raise MissingAPIKeyError(
               "Freesound API key not found.\n"
               "Set FREESOUND_API_KEY environment variable"
          )
     
     return key


class FreesoundAudioDownloader(AudioDownloader):
    '''
    AudioDownloader backend implementation which uses
    the freesound API to https://freesound.org/docs/api/
    download audio samples
    '''

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

        api_key = get_freesound_api_key()
        self.client.set_token(api_key, "token")

    
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

    def download_audio(self, query):
            '''
            :param query: string to use as the query when calling the API

            :return paths of the downloaded files in a list
            '''
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
                