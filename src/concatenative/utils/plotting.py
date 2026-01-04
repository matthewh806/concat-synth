import matplotlib.pyplot as plt
import numpy as np
from typing import List
from concatenative.core import AudioSnippet
from concatenative.core.features import FEATURE_MAP
import logging

logger = logging.getLogger(__name__)

class InteractiveCorpusPlot:
    '''
    Creates an interactive 3 dimensional scatter plot by mapping features to x, y & colour
    The snippets should be normalised so that the values for each feature
    are in the range [0, 1]
    
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
            title: str = "Corpus Normalised Feature Map"
    ):
        self.snippets = snippets

        features_to_plot = [x_axis_feature, y_axis_feature, colour_feature]
        plot_data = {feat_name: [] for feat_name in features_to_plot}

        for snippet in snippets:
            for feature in features_to_plot:
                if feature in snippet.normalised_features:
                    plot_data[feature].append(snippet.normalised_features[feature])
                else:
                    logger.warning("Feature {feature} missing from {snippet}. Excluding from plot")

        self.x_data = np.array(plot_data[x_axis_feature])
        self.y_data = np.array(plot_data[y_axis_feature])
        self.colour_data = np.array(plot_data[colour_feature])
        self.title = title

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

        # -- Step 6: Connect event handlers
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

        self.annot.set_visible(False)

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