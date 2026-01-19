import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Callable, Dict
from concatenative.audio.audio_snippet import AudioSnippet
from concatenative.path.concatenation_path import ConcatenationPath
from concatenative.analysis.corpus import Corpus
from uuid import UUID
import logging

logger = logging.getLogger(__name__)

class InteractiveCorpusPlot:
    '''
    Creates an interactive 3 dimensional scatter plot by mapping features to x, y & colour
    The snippets should be normalised so that the values for each feature
    are in the range [0, 1]
    
    #TODO Change to take Corpus as an arg instead of snippets

    :param snippets: A list of analysed AudioSnippets
    :param x_axis_feature: The name of the feature for the x axis
    :param y_axis_feature: The name of the feature for the y axis
    :param colour_feature: The name of the feature for the colour dimension
    :param title: The title for the plot
    '''
        
    def __init__(
            self,
            snippets: List[AudioSnippet],
            x_axis_feature: str,
            y_axis_feature: str,
            colour_feature: str,
            title: str = "Corpus Normalised Feature Map",
            on_click_callback: Optional[Callable[[AudioSnippet], None]] = None,
            path_to_draw: Optional[ConcatenationPath] = None
    ):
        self.snippets = snippets

        features_to_plot = [x_axis_feature, y_axis_feature, colour_feature]
        plot_data = {feat_name: [] for feat_name in features_to_plot}

        for snippet in snippets:
            for feature in features_to_plot:
                if feature in snippet.normalised_features:
                    plot_data[feature].append(snippet.normalised_features[feature])
                else:
                    logger.warning(f"Feature {feature} missing from {snippet}. Excluding from plot")

        self.x_data = np.array(plot_data[x_axis_feature])
        self.y_data = np.array(plot_data[y_axis_feature])
        self.colour_data = np.array(plot_data[colour_feature])
        self.title = title
        self.on_click_callback = on_click_callback
        self.path_to_draw = path_to_draw

        self.snippet_id_to_index : Dict[UUID, int] = {snippet.id: index for index, snippet in enumerate(self.snippets)}

        self.fig, self.ax = plt.subplots(figsize=(12,10))
        self.scatter = self.ax.scatter(
            x = self.x_data,
            y = self.y_data,
            c = self.colour_data,
            cmap='viridis'
        )

        # --- Step 4: Add labels and a colour bar for clarity --- 
        self.ax.set_title(title, fontsize=16)
        self.ax.set_xlabel(f"{x_axis_feature}", fontsize=12)
        self.ax.set_ylabel(f"{y_axis_feature}", fontsize=12)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        cbar = self.fig.colorbar(self.scatter, ax=self.ax)
        cbar.set_label(f"Color: {colour_feature}", fontsize=12)

        # -- Step 5: Add an annotation object to display Snippet details
        self.annot = self.ax.annotate(
            "", xy=(0,0), xytext=(15,15), textcoords='offset points'
        )

        self.annot.set_visible(False)

        # -- Step 6: Connect event handlers
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

        # -- Step 7: Draw the audio path (if provided)
        if self.path_to_draw:
            self.draw_path()


    def update_annot(self, index):
        '''
        Updates the annotations text and position.
        '''
        pos = self.scatter.get_offsets()[index]
        self.annot.xy = pos

        snippet = self.snippets[index]

        feature_filename = snippet.metadata['filename'] if snippet.metadata['filename'] else ""
        feature_txt = "\n".join([f"{k}: {v:.2f}" for k, v in snippet.features.items()])

        self.annot.set_text(f"ID: {snippet.id}\nfilename: {feature_filename}\n{feature_txt}")


    def hover(self, event):
        '''
        Handles mouse hover events (motion_notify_event)
        
        :param event: associated event (MouseEvent)
        '''

        is_visible = self.annot.get_visible()
        if event.inaxes == self.ax:
            contains, details = self.scatter.contains(event)
            if contains:
                # Gets the index of the point under the mouse
                index = details['ind'][0]
                self.update_annot(index)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if is_visible:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()


    def on_press(self, event):
        '''
        Handles button press events (button_press_event)
        :param event: associated event (MouseEvent)
        '''

        if event.inaxes == self.ax:
            contains, details = self.scatter.contains(event)
            logger.debug(f"OnPress: {contains}, {details}")

            if contains:
                # Get the index of the point under the mouse. 
                # Just get the first by default 
                index = details['ind'][0]
                snippet = self.snippets[index]

                if self.on_click_callback:
                    self.on_click_callback(snippet)

    def draw_path(self):
        if not self.path_to_draw:
            logger.warning("Can't draw None path object")
            return
        
        if len(self.path_to_draw) < 2:
            logger.warning(f"Not enough points ({len(self.path_to_draw)}) in the path to plot")
            return

        # Get the indicies of the paths we're plotting
        path_indices = [self.snippet_id_to_index[snippet.id] for snippet in self.path_to_draw.snippets_path]

        # Just in case the snippet from the path to draw wasn't in the map (i.e. not in the plotted data)
        valid_indices = [i for i in path_indices if i is not None]

        # Get the x and y values of each of the points
        path_x = self.x_data[valid_indices]
        path_y = self.y_data[valid_indices]

        self.ax.plot(
            path_x, path_y, 
            color='red', linestyle='-', linewidth=1.5, 
            alpha=0.8, label='Concatenation Path')
        
        self.ax.plot(path_x[0], path_y[0],
                     marker='o', color='lime', markersize=10,
                     alpha=0.9, label='start')
        
        self.ax.plot(path_x[-1], path_y[-1],
                marker='X', color='red', markersize=10,
                alpha=0.9, label='end')

        self.ax.legend()


def plot_corpus_feature_distribution(corpus: Corpus, feature_name: str, bins: int = 30):
    '''
    Creates and plots a histogram for a single features distribution from a Corpus
    
    :param corpus: A fully analysed Corpus instance
    :param feature_name: The name of the feature to plot (e.g. 'rms')
    :param bins: The number of bins for the histogram
    '''

    if not corpus.snippets:
        logger.warning("Corpus has no snippets to plot")
        return

    # Get the feature values out of the corpus
    # TODO Make this part of the corpus
    feature_values = []
    for snippet in corpus.snippets:
        if feature_name in snippet.features.keys():
            feature_values.append(snippet.features[feature_name])
        else:
            logger.warning(f"Snippet {snippet.id} is missing feature '{feature_name}")

            
    if not feature_values:
        print(f"No valid data for feature '{feature_name} to plot")
        return

    
    _, ax = plt.subplots(figsize=(12,6))

    ax.hist(feature_values, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of '{feature_name}' in Corpus of size {len(corpus)} samples")
    ax.set_xlabel(f"{feature_name} value")
    ax.set_ylabel(f"Number of snippets")

    plt.grid(axis='y', alpha=0.75)
    plt.show()


def plot_signal_segmentation(samples: np.ndarray, 
                             segments: List[np.ndarray], 
                             segment_colors: list[str] = ['skyblue', 'lightcoral', 'lightgreen', 'plum'],
                             ):
    
    '''
    Quick and dirty segmentation plot

    # TODO
    Needs improving a lot! 
    It would be better to provide the segmentation times rather than the segments themselves...?
    The segment values arent used, just the length of each. 
    We assume segments start at 0 and are continuguous, otherwise it doesn't work.
    '''
    
    if len(samples) == 0:
        logger.warning("No signal to plot")
        return
    
    if len(segments) == 0:
        logger.warning("No segments to plot")
        return
    
    _, ax = plt.subplots(figsize=(12,6))
    ax.plot(samples)

    # What about segments which don't start at 0?
    start = 0
    for idx, segment in enumerate(segments):
        end = start + len(segment)
        ax.axvspan(start, end, alpha=0.7, color = segment_colors[idx % len(segment_colors)])
        start = end

    ax.set_title(f"Segmentation of a signal")
    ax.set_xlabel(f"Sample")
    ax.set_ylabel(f"Amplitude")

    plt.grid()
    plt.show()


