from abc import ABC, abstractmethod
from pathlib import Path

class AudioDownloader(ABC):
    '''
    Abstract class definining an interface for the 
    audio downloader subclasses
    '''
    @abstractmethod
    def download_audio(self, query) -> list[Path]:
        pass
