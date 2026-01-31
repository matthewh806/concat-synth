## [0.5.0] - 31-01-2026

### Added
- **Configurations file:** This allows finer grain control of under the hood settings not exposed by the CLI
- **Units to the Feature class:** This helps with plotting the values to get more context
- **Feature Weights:** Customisable weights per audio feature allow much more precise tuning of the nearest neighbour calculation
    * Configurable from the command line
- **Visualisation of feature trajectory:** Allows tracking of a feature in time 
- **Audio Segmentation:** The input audio is segmented into individual smaller snippets, instead of a single fixed length slice
    * There are three strategies for splitting up the audio: `onsets`, `slices` (fixed size), `none` (uses the whole file) 
    * The strategy and its parameters are configurable from the command line
- **Automatic plotting:** Using the `--plot` CLI flag will optionally generate and save all of the visualisation plots available for the last run

### Fixed
- **Incorrect `ConcatenationPath` member name:** `cross_fade_seconds` was renamed to `cross_fade_milliseconds`

## [0.4.0] - 17-01-2026

### Added
- **Configurable Feature Extraction:** The features to be analysed are now configurable from the command line
    * Implemented a `Feature Registry` to define available audio features (e.g. rms, pitch).
    * Added a `--features` CLI argument allows users to specify a comma-separated list of features to use for a given run.
- **Testing Framework (`pytest`):** Integrated the pytest framework into the project for unit testing.
- **Corpus Feature Distribution Plots:** New visualization function (`plot_corpus_feature_distribution`) to generate histogram feature plots, 
    * Allows for visual analysis of how features are distributed across the entire corpus.

### Changed
- **Parallel audio analyses:** The feature extraction happens across multiple different processes simultaneously, cutting down on the overall time taken
    * Added generic `run_parallel_cpu_tasks` & `
- **Wav audio format support:** In addition to `mp3` input formats, `wav` is also supported
    * Adding further formats is as simple as modifying `SUPPORTED_AUDIO_EXTENSIONS` in `constants.py`

### Fixed
- **Pitch detection bug:** Librosa was failing to an overall average pitch for a signal if there were any unvoiced (`NaN`) frames. 
    * These unvoiced frames are simply skipped over in calculating the average using `np.nanmean`


## [0.3.0] - 06-01-2026

### Added

- **Interactive Corpus Visualization:** Introduced a new visualization module with an `InteractiveCorpusPlot` class. This provides a 2D scatter plot of the corpus feature space with capabilities for:
    * Hovering over points to view snippet details.
    * Clicking points to trigger a callback (e.g., for audio playback).
    * Drawing the generated concatenation path on the plot.
- **`ConcatenationPath` Class:** The generated path is now a dedicated ConcatenationPath object instead of a simple list. This new class encapsulates path metadata, generation parameters, and behavior like rendering and
statistics generation.
- **`Corpus` Class:** This class holds the entire collection of AudioSnippets and the analysis results
- **`requirements.txt`:** For installing all dependencies via pip

### Changed

- **Nearest Neighbour Algorithm**: This is no longer calculated on the fly when generating the path, but pre-calculated and stored in a `scipy.spatial.
KDTree`
- **Major Architectural Refactoring**: The core application has been reorganized for better clarity, scalability, and maintainability.
    * The generic core directory has been broken down into domain-specific packages: audio, analysis, and selection.
- **Prevent Repeat YouTube Downloads**: The YouTube downloader will no longer download the same video each time for a given query. 

### Fixed
- **Incorrect Path Generation:** The exclusion list was being compared against the Snippet itself rather than the ID, meaning the result was always false. This led to paths being generated which just oscillated between two nearest neighbour snippets


## [0.2.0] - 01-01-2026

### Added

- **Feature-based Synthesis Engine:** A new synthesis engine that creates structured audio by finding the "nearest neighbor" in a multi-dimensional feature space, replacing the previous random selection.
- **Audio Feature Analysis:** An analysis pipeline to extract key audio features (e.g., RMS loudness, pitch, spectral centroid) from all audio snippets before synthesis.
- **Corpus-Wide Feature Normalization:** Implemented Min-Max scaling for all features across the entire corpus, ensuring fair contribution from each feature in distance calculations.
- **Non-Repetitive Generation Logic:** A "recently used" history mechanism (`collections.deque`) to prevent the synthesizer from immediately re-selecting the same snippets, increasing output variety.
- **Structured Logging:** A robust logging system using Python's built-in `logging` module, with configurable verbosity (`--verbose`) 
- **Performance Measurement Tools:** A `@timed` decorator to easily measure and log function execution times for identifying performance bottlenecks.
- **Data-Driven Feature Configuration:** Refactored feature definitions into a declarative `FEATURE_MAP`, making it easier to add, remove, or modify the audio features used by the synthesizer.

### Changed

- **Core Synthesis Algorithm:** The fundamental logic for audio generation has been changed from a random shuffle to a deterministic pathfinding algorithm based on audio feature similarity.

### Fixed

- **Inaccurate Output Duration:** The path generation logic now correctly accounts for the cross-fade duration, resulting in an output duration that more accurately matches the users target.
- **Target Length Overshoot:** The final rendered audio is now precisely trimmed to the requested output length, preventing the generated file from being longer than specified.

## [0.1.0] - 30-12-2025

### Added
- **Initial Project Setup:** Established the project as a standard Python package, including basic directory structure and dependency management.
- **Basic Command-Line Interface (CLI):** Implemented a basic CLI allowing users to run synthesis from a local directory or online sources.
- **Core Audio Processing:** Functionality to load audio segments from disk and concatenate them into a single output file.
- **Multiple Audio Sources:** Support for sourcing audio snippets from local file directories, Freesound, and YouTube.
- **Configurable Synthesis Parameters:** Users can adjust basic synthesis parameters such as maximum audio snippet length and crossfade duration.
- **Parallel Audio Downloading:** Implemented concurrent downloading of audio snippets to improve data collection speed.
- **Random Snippet Selection:** The initial synthesis approach involved randomly selecting and concatenating audio snippets.
