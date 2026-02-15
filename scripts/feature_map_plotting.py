from pathlib import Path
import matplotlib.pyplot as plt
import sounddevice as sd
from concatenative.visualisation.plotting import InteractiveCorpusPlot
from common import setup_and_run_synthesis, get_feature_set, get_parser
import logging
import sys

def play_snippet_audio_callback(snippet):
    '''
    Callback method which is passed to InteractiveCorpusPlot
    Will be called when a point in the scatter plot is clicked
    And uses sounddevice to playback the snippets audio
    
    :param snippet: AudioSnippet to playback
    '''

    if not snippet:
        return
    
    try:
        sd.stop() 
        sd.play(snippet.samples, snippet.sample_rate, blocking=False)
    except Exception as e:
        print(f"Error playing audio: {e}")


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    corpus, concat_path, config = setup_and_run_synthesis(args.config)
    print(concat_path.get_stats())    
    feature_set = get_feature_set(config)

    if len(feature_set) < 3:
        logging.error(f"Need at least 3 features to plot the feature map, got {len(feature_set)}")
        sys.exit(1)

    plot = InteractiveCorpusPlot(corpus.snippets, 
                                 feature_set[0], 
                                 feature_set[1], 
                                 feature_set[2], 
                                 normalised=False,
                                 on_click_callback=play_snippet_audio_callback, 
                                 path_to_draw=concat_path)
    plt.show()