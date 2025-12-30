# Concatenative Synthesizer

  A command-line tool for creating new sounds through concatenative synthesis. It works by taking many small audio snippets and stitching them together with crossfades to generate new audio textures.

  ## Features

   - Multiple Sources: Generate audio from a local directory of files or by downloading snippets from Freesound or YouTube.
   - Configurable Synthesis: Adjust the maximum length of audio slices and the duration of the crossfade between them.
   - Intelligent Downloading: Uses individual words for precise searching on Freesound and randomly generated phrases for creative and unpredictable results from YouTube.
   - Parallel Processing: Downloads audio snippets in parallel for faster data collection.

  ## Requirements

   - Python 3.9+
   - A [Freesound](https://freesound.org/home/app_new/) API Key (only required for the freesound backend).

  ## Installation

   1. Clone the repository:

   ```
    git clone https://github.com/matthewh806/concatenative.git
    cd concatenative
   ```
      

   2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   3. Install the project in editable mode:
      This command will install the necessary dependencies and make the concat-synth command available in your environment.
   
   ```
   pip install -e .
   ```

  ## Configuration

  For the Freesound backend to work, you must set your API key as an environment variable.

  `export FREESOUND_API_KEY="your_api_key_here"`

  You can add this line to your shell's startup file (e.g., .zshrc or .bash_profile) to make it permanent.

  ## Usage

  The primary entry point is the `concat-synth` command, which has two main sub-commands: `dir` and `download`.

  ### Generate from a local directory

  The dir command recursively finds all audio files in a given directory and uses them for synthesis.

```
concat-synth dir path/to/audio --out my_sound.wav --fade 100 --max-slice-length 0.2
```

  ### Generate by downloading audio

  The download command fetches audio from either freesound or youtube based on a list of words.

  Freesound Example:

   ```
   concat-synth download freesound --words words.txt --max-snippets 64 --out freesound_mix.wav
   ```

  YouTube Example:
```
concat-synth download youtube --words words.txt --max-snippets 32 --out youtube_mix.wav
```

  Command-line Options

   - `--out <path>`: Path to save the output .wav file. (Default: output.wav)
   - `--max-slice-length` <seconds>: Maximum length of each audio snippet. (Default: 0.5)
   - `--fade <ms>`: Duration of the crossfade in milliseconds. (Default: 50)
   - `--words <path>`: Path to a text file containing words to use as search terms. (Default: words.txt)
   - `--max-snippets <int>`: The maximum number of audio snippets to download and use. (Default: 32)