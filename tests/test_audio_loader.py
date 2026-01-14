from concatenative.audio.audio_loader import find_audio_files_recursively
from pathlib import Path
import pytest

def test_find_files_in_flat_directory(tmp_path: Path):
    '''
    Test that the function finds all supported audio files in a flat directory structure
    '''

    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    (tmp_path / "c.mp3").touch()
    (tmp_path / "d.txt").touch()

    found_files = set(find_audio_files_recursively(tmp_path))

    assert len(found_files) == 3
    expected_files = {tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.mp3"}
    assert found_files == expected_files


def test_find_files_recursively(tmp_path: Path):
    '''
    Test that the function finds all supported audio files in a nested directory structure
    '''

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    drums_dir = samples_dir / "drums"
    drums_dir.mkdir()

    analysis_dir = samples_dir / "analysis"
    analysis_dir.mkdir()

    (tmp_path / "song.wav").touch()
    (samples_dir / "drums.mp3").touch()
    (drums_dir / "kick.mp3").touch()
    (drums_dir / "snare.wav").touch()
    (analysis_dir / "kick.json").touch()
    (analysis_dir / "snare.json").touch()

    found_files = set(find_audio_files_recursively(tmp_path))
    expected_files = {tmp_path / "song.wav", samples_dir / "drums.mp3", drums_dir / "kick.mp3", drums_dir / "snare.wav"}
    assert found_files == expected_files


def test_find_files_with_custom_extensions(tmp_path: Path):
    '''
    Test that the providing supported custom extensions finds files
    '''

    (tmp_path / "a.mp3").touch()
    (tmp_path / "a.wav").touch()

    found_files = set(find_audio_files_recursively(tmp_path, extensions={'.mp3'}))
    assert found_files == { tmp_path / "a.mp3" }


def test_error_is_raised_when_unsupported_extensions_provided(tmp_path: Path):
    '''
    Test that an error is raised when unsupported extensions are provided
    '''

    (tmp_path / "a.mp3").touch()
    (tmp_path / "a.wav").touch()

    with pytest.raises(ValueError):
        find_audio_files_recursively(tmp_path, extensions={'.txt'})


def test_raises_error_for_invalid_path(tmp_path: Path):
    '''
    Tests that an error is raised if the path provided is not a directory
    '''

    file_path = tmp_path / "not_a_directory.txt"
    file_path.touch

    with pytest.raises(ValueError):
        find_audio_files_recursively(file_path)

    non_existent_path = tmp_path / "does_not_exist"

    with pytest.raises(ValueError):
        find_audio_files_recursively(non_existent_path)