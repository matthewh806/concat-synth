from pathlib import Path
from concatenative.visualisation.plotting import plot_corpus_feature_distribution
from common import setup_and_run_synthesis, get_feature_set, get_parser

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    corpus, concat_path, config = setup_and_run_synthesis(args.config)
    print(concat_path.get_stats())    
    feature_set = get_feature_set(config)

    for feature in feature_set:
        plot_corpus_feature_distribution(corpus, feature=feature, bins=100)