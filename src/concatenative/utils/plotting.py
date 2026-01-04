import matplotlib.pyplot as plt
import numpy as np
from typing import List
from concatenative.core import AudioSnippet
from concatenative.core.features import FEATURE_MAP
import logging

logger = logging.getLogger(__name__)

def plot_feature_map(
        snippets: List[AudioSnippet],
        x_axis_feature: str,
        y_axis_feature: str,
        colour_feature: str,
        title: str = "Corpus Normalised Feature Map"
):
    '''
    Creates a 3 dimensional scatter plot by mapping features to x, y & colour
    The snippets should be normalised so that the values for each feature
    are in the range [0, 1]
    
    :param snippets: A list of analysed AudioSnippets
    :param x_axis_feature: The name of the feature for the x axis
    :param y_axis_feature: The name of the feature for the y axis
    :param colour_feature: The name of the feature for the colour dimension
    :param title: The title for the plot
    '''
    # --- Step 1: Extract normalised feature data from snippets ---
    features_to_plot = [x_axis_feature, y_axis_feature, colour_feature]
    plot_data = {feat_name: [] for feat_name in features_to_plot}

    for snippet in snippets:
        for feature in features_to_plot:
            if feature in snippet.normalised_features:
                plot_data[feature].append(snippet.normalised_features[feature])
            else:
                logger.warning("Feature {feature} missing from {snippet}. Excluding from plot")

    if not all(plot_data.values()):
        logger.warning("Could not generate plot. Not enough valid data for the selected features")
        return
    
    print(plot_data)

    # --- Step 2: Convert feature lists to numpy arrays ---
    x_data = np.array(plot_data[x_axis_feature])
    y_data = np.array(plot_data[y_axis_feature])
    colour_data = np.array(plot_data[colour_feature])

    # --- Step 3: Create the scatter plots for X and Y features ---
    fig, ax = plt.subplots(figsize=(12,10))

    scatter = ax.scatter(
        x = x_data,
        y = y_data,
        c = colour_data,
        cmap='viridis'
    )

    # --- Step 4: Add labels and a colour bar for clarity --- 
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f"{x_axis_feature}", fontsize=12)
    ax.set_ylabel(f"{y_axis_feature}", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f"Color: {colour_feature}", fontsize=12)

    plt.show()