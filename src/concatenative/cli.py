import argparse
import logging
from .concatenative_synth import run_download_backend, run_dir_backend
from concatenative.utils import setup_logger
from concatenative.analysis.available_features import FEATURE_REGISTRY
from concatenative.analysis.segmentation import SEGMENTATION_MAP
from concatenative.constants import SUPPORTED_AUDIO_EXTENSIONS

'''
This is a script which defines the CLI for the concatenative synthesis system.
From the installed package it can be run as: concat-synth
'''

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--out", type=str, default="output.wav",
        help=(
            "Output audio file path. Must include the file extension.\n"
            f"Available options: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    )
    parent_parser.add_argument(
        "--output-length", type=float, default=10.0,
        help="Length of the concatenated output file"
    )
    parent_parser.add_argument(
        "--max-slice-length", type=float, default=0.5,
        help="Maximum length of each slice (seconds)"
    )
    parent_parser.add_argument(
        "--features", type=str, default='rms, spectral centroid, pitch',
        help=(  
            "A comma separated list of features to use for analysis.\n"
            f"Available options: {', '.join(FEATURE_REGISTRY.keys())}"
        )
    )
    parent_parser.add_argument(
        "--weight", type=str, default=None,
        help=(
            "A comma separated list of feature weight floats to use for the distance calculations\n"
            "Should match the order and length of the features provided above\n (e.g. '0.75, 0.3, 0.2')"
        )
    )
    parent_parser.add_argument(
        "--segmentation", type=str, default='none',
        help=(
            "The strategy to use to slice up individual audio samples.\n"
            f"Available options: {', '.join(SEGMENTATION_MAP.keys())}"
        )
    )
    parent_parser.add_argument(
        "--max-sample-slices", type=int, default=None,
        help=(
            "The maximum number of slices to generate via segmentation per sample.\n"
            "Prevents the corpus growing too large"
        )
    )
    parent_parser.add_argument(
        "--fade", type=int, default=50,
        help="Cross fade length (milliseconds)"
    )
    parent_parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate plots for the corpus and concatenation"
    )
    parent_parser.add_argument(
        "-v", '--verbose',
        help="Enable verbose logging (DEBUG level).",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO
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
        "--words", type=str, default="data/words.txt",
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
    
    if not args.command:
        parser.print_help()
        exit(0)
    
    setup_logger(log_level=args.loglevel)

    feature_list = args.features.split(",")
    features = []
    for feature_name in feature_list:
        feature_name = feature_name.lstrip().rstrip()
        if not feature_name in FEATURE_REGISTRY.keys():
            raise ValueError(f"Feature {feature_name} not a known feature, available features: {', '.join(FEATURE_REGISTRY.keys())}")
        features.append(feature_name)

    feature_weight_list = args.weight.split(",") if args.weight else None
    feature_weights = {}
    if feature_weight_list and len(feature_weight_list) != len(features):
        raise ValueError(f"Feature weights list ({len(feature_weight_list)}) must be the same length as the feature list {len(feature_list)}")
    
    if feature_weight_list:
        for idx, weight in enumerate(feature_weight_list):
            try:
                weight_f = float(weight.lstrip().rstrip())
            except ValueError:
                raise ValueError(f"Feature weight {weight} must be of type float, got {type(weight_f)}")
            
            if weight_f < 0.0:
                raise ValueError(f"Feature weight must be a positive value, got {weight_f}")

            feature_weights[features[idx]] = weight_f


    if args.command == "download":
        run_download_backend(
            backend_name = args.backend,
            words_path = args.words,
            output_length=args.output_length,
            output_path = args.out,
            feature_set=features,
            feature_weights = feature_weights,
            max_snippets = args.max_snippets,
            max_snippet_length=args.max_slice_length,
            max_slices_per_sample=args.max_sample_slices,
            cross_fade=args.fade,
            segmentation_strategy=args.segmentation,
            plots=args.plot
        )
    elif args.command == "dir":
        run_dir_backend(
            input_dir= args.input_dir,
            output_path= args.out,
            feature_set=features,
            feature_weights = feature_weights,
            output_length=args.output_length,
            max_snippet_length=args.max_slice_length,
            max_slices_per_sample=args.max_sample_slices,
            cross_fade=args.fade,
            segmentation_strategy=args.segmentation,
            plots=args.plot
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()