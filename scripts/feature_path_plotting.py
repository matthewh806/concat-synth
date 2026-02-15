from pathlib import Path
from common import setup_and_run_synthesis, get_feature_set, get_parser
from concatenative.visualisation.plotting import plot_feature_vs_time

'''
Plots a feature value over time in a ConcatenationPath
'''

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    corpus, concat_path, config = setup_and_run_synthesis(args.config)
    concatenated_signal = concat_path.render(output_length=30, output_sr=44100)
    feature_set = get_feature_set(config)
    
    for feature in feature_set:
        plot_feature_vs_time(concatenated_signal, concat_path, feature)


