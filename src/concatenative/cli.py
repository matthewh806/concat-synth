import argparse
from concatenative_synth import run_download_backend, run_dir_backend

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--out", type=str, default="output.wav",
        help="Output WAV file path"
    )
    parent_parser.add_argument(
        "--max-slice-length", type=float, default=0.5,
        help="Maximum length of each slice (seconds)"
    )
    parent_parser.add_argument(
        "--fade", type=int, default=50,
        help="Cross fade length (milliseconds)"
    )

    parser = argparse.ArgumentParser("Concatenative Audio Synthesis")
    subparsers = parser.add_subparsers(dest="command")

    #---------------------------------
    # Subcommand: Download and concat
    #---------------------------------
    download_parser = subparsers.add_parser("download", parents=[parent_parser], help="Download audio from backend and concatenate")
    download_parser.add_argument(
        "backend", choices=["youtube", "freesound"],
        help="Backend to use for downloading audio"
    )
    download_parser.add_argument(
        "--words", type=str, default="words.txt",
        help="Path to the word list for phrase generation"
    )

    download_parser.add_argument(
        "--max-snippets", type=int, default=32,
        help="Max number of snippets to download"
    )

    #---------------------------------
    # Subcommand: Use local files
    #---------------------------------
    dir_parser = subparsers.add_parser("dir", parents=[parent_parser], help="Use existing audio files from directory")
    dir_parser.add_argument("input_dir", type=str, help="Directory containing audio files")

    args = parser.parse_args()
    if args.command == "download":
        run_download_backend(
            backend_name = args.backend,
            words_path = args.words,
            output_path = args.out,
            max_snippets = args.max_snippets,
            max_snippet_length=args.max_slice_length,
            cross_fade=args.fade
        )
    elif args.command == "dir":
        run_dir_backend(
            input_dir= args.input_dir,
            output_path= args.out,
            max_snippet_length=args.max_slice_length,
            cross_fade=args.fade
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()