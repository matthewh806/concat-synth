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
