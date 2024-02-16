"""
summary
"""

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def visualize_confusion_matrix(
    y_true,
    y_pred,
    classes: list,
    fig_size: tuple = (8, 6),
    color_bar: bool = False,
    title: str = "Confusion Matrix",
):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_
        fig_size (tuple, optional): _description_. Defaults to (8, 6).
        color_bar (bool, optional): _description_. Defaults to False.
        title (str, optional): _description_. Defaults to "Confusion Matrix".

    Returns:
        _type_: _description_
    """
    # Calculate confusion matrix values & normalized values
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    # Calculate the confusion matrix in percentages of total (for text color)
    cm_pct = cm / cm.sum()

    # Confusion matrix display objects
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Create the figure object and axis
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()

    # Plot confusion matrix on the axis
    disp.plot(ax=ax, cmap="Blues", colorbar=color_bar)

    # Function to dynamically generate text colors
    def text_color(cmap_name: str, frac: float) -> str:
        cmap = plt.get_cmap(cmap_name)
        colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]
        target_idx = int(frac * len(colors))
        return colors[target_idx]

    # Annotate confusion matrix with percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            # Fetch percentage value & appropriate text color
            value = cm_norm[i, j]
            hex_color = text_color("Blues_r", cm_pct[i, j])

            # Add text in the plot
            ax.text(
                j,
                i + 0.1,
                f"{value:.2%}",
                ha="center",
                va="center",
                color=hex_color,
            )

    # Plot customization
    ax.set_title(title, fontdict={"weight": "bold"})

    # Show plot
    plt.show()
