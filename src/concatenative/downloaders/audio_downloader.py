from abc import ABC, abstractmethod
from pathlib import Path

class AudioDownloader(ABC):
    '''
    Abstract class definining an interface for the 
    audio downloader subclasses
    '''
    @abstractmethod
    def get_snippets(self, query) -> list[Path]:
        pass
