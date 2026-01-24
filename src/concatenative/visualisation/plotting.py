import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Callable, Dict
from concatenative.audio.audio_snippet import AudioSnippet
from concatenative.path.concatenation_path import ConcatenationPath
from concatenative.analysis.corpus import Corpus
from concatenative.analysis.features import Feature
from uuid import UUID
import logging

logger = logging.getLogger(__name__)

def get_feature_label(feature: Feature, normalised):
    label =  f"{feature.name}"
    if not normalised and feature.units:
        label = label + f" ({feature.units})"

    return label

class InteractiveCorpusPlot:
    '''
    Creates an interactive 3 dimensional scatter plot by mapping features to x, y & colour
    The snippets should be normalised so that the values for each feature
    are in the range [0, 1]
    
    #TODO Change to take Corpus as an arg instead of snippets

    :param snippets: A list of analysed AudioSnippets
    :param x_axis_feature: The feature for the x axis
    :param y_axis_feature: The feature for the y axis
    :param colour_feature: The feature for the colour dimension
    :param normalised: Plot the normalised values
    '''
        
    def __init__(
            self,
            snippets: List[AudioSnippet],
            x_axis_feature: Feature,
            y_axis_feature: Feature,
            colour_feature: Feature,
            normalised = True,
            on_click_callback: Optional[Callable[[AudioSnippet], None]] = None,
            path_to_draw: Optional[ConcatenationPath] = None
    ):
        self.snippets = snippets

        features_to_plot = [x_axis_feature.name, y_axis_feature.name, colour_feature.name]
        plot_data = {feat_name: [] for feat_name in features_to_plot}

        for snippet in snippets:
            for feature in features_to_plot:
                if feature in snippet.normalised_features:
                    plot_data[feature].append(snippet.normalised_features[feature] if normalised else snippet.features[feature])
                else:
                    logger.warning(f"Feature {feature} missing from {snippet}. Excluding from plot")

        self.x_data = np.array(plot_data[x_axis_feature.name])
        self.y_data = np.array(plot_data[y_axis_feature.name])
        self.colour_data = np.array(plot_data[colour_feature.name])
        self.title = "Corpus Normalised Feature Map" if normalised else "Corpus Feature Map"
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

        x_label = get_feature_label(x_axis_feature, normalised)
        y_label = get_feature_label(y_axis_feature, normalised)
        color_label = get_feature_label(colour_feature, normalised)


        # --- Step 4: Add labels and a colour bar for clarity --- 
        self.ax.set_title(self.title, fontsize=16)
        self.ax.set_xlabel(x_label, fontsize=12)
        self.ax.set_ylabel(y_label, fontsize=12)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        cbar = self.fig.colorbar(self.scatter, ax=self.ax)
        cbar.set_label(f"Color: {color_label}", fontsize=12)

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


def plot_corpus_feature_distribution(corpus: Corpus, feature: Feature, bins: int = 30):
    '''
    Creates and plots a histogram for a single features distribution from a Corpus
    
    :param corpus: A fully analysed Corpus instance
    :param feature: The feature to plot
    :param bins: The number of bins for the histogram
    '''

    if not corpus.snippets:
        logger.warning("Corpus has no snippets to plot")
        return

    # Get the feature values out of the corpus
    # TODO Make this part of the corpus
    feature_values = []
    feature_name = feature.name
    feature_units = feature.units
    for snippet in corpus.snippets:
        if feature.name in snippet.features.keys():
            feature_values.append(snippet.features[feature_name])
        else:
            logger.warning(f"Snippet {snippet.id} is missing feature '{feature_name}")

            
    if not feature_values:
        print(f"No valid data for feature '{feature_name} to plot")
        return

    
    _, ax = plt.subplots(figsize=(12,6))

    ax.hist(feature_values, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of '{feature_name}' in Corpus of size {len(corpus)} samples")
    ax.set_xlabel(f"{feature_name} value {feature_units if feature_units else ""}")
    ax.set_ylabel(f"Number of snippets")

    plt.grid(axis='y', alpha=0.75)
    plt.show()


def plot_signal_segmentation(samples: np.ndarray, 
                             segments: tuple[np.ndarray, int, int], 
                             segment_colors: list[str] = ['skyblue', 'lightcoral', 'lightgreen', 'plum'],
                             ):
    
    '''
    Plots the signal and the segmentation slices on top of it

    :param samples the signal to plot
    :param segments the segment data for the signal (segment samples, start sample, end sample)
    :param segment_colors colours to cycle through when plotting the segements (to aid visualisation)
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
    for idx, segment in enumerate(segments):
        start = segment[1]
        end = segment[2]
        ax.axvspan(start, end, alpha=0.7, color = segment_colors[idx % len(segment_colors)])
        start = end

    ax.set_title(f"Segmentation of a signal")
    ax.set_xlabel(f"Sample")
    ax.set_ylabel(f"Amplitude")

    plt.grid()
    plt.show()


def plot_feature_vs_time(path : ConcatenationPath, feature: Feature):
    '''
    Creates a plot of how a specific feature changes across
    time in a ConcatenationPath

    :param path: The concatenation path to plot the feature from
    :param feature: The feature to plot
    '''

    if len(path) == 0:
        logger.warning("Empty path provided, nothing to plot!")
        return

    # Get the sample rate out of the first snippet... assume they're all the same
    sr = path.snippets_path[0].sample_rate

    if sr == 0:
        raise ValueError(F"Sample rate value is 0. Invalid!")

    # Get the feature out of the path
    # TODO Should a concatenation path be able to provide all of this information itself?
    feature_values = []
    cross_fade = int(path.cross_fade_seconds / 1000 * sr)
    snippet_sample_positions = []
    running_sample_position = 0
    for snippet in path.snippets_path:
        if feature.name not in snippet.features:
            raise ValueError(f"Feature {feature.name} not found in snippet {snippet}")
        
        feature_value = snippet.features[feature.name]
        feature_values.append(feature_value)

        snippet_sample_positions.append(running_sample_position)
        running_sample_position += (len(snippet.samples) - cross_fade)

    # Get the length of the path in seconds
    snippet_start_times = [snippet_start_sample / sr for snippet_start_sample in snippet_sample_positions]

    _, ax = plt.subplots(figsize=(12,6))
    ax.step(snippet_start_times, feature_values, where='post')

    y_label = get_feature_label(feature, normalised=False)
    ax.set_title(f"Feature {feature.name} in concatenated signal vs time")
    ax.set_xlabel(f"Time (s)")
    ax.set_ylabel(y_label)

    plt.grid()
    plt.show()