from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from skimage import img_as_ubyte
from torch.utils.data import Dataset


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: list, figsize: Tuple[int, int] = (10, 8), fontsize: int = 14
) -> Figure:
    """Plots a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Args:
        confusion_matrix: The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
                            Similarly, constructed ndarrays can also be used.
        class_names: An ordered list of class names, in the order they index the given confusion matrix.
        figsize:  A 2-long tuple, the first value determining the horizontal size of the outputted figure,
                    the second determining the vertical size. Defaults to (10,7).
        fontsize: Font size for axes labels. Defaults to 14.

    Returns:
        The resulting confusion matrix figure.
    """
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError as ve:
        raise ValueError("Confusion matrix values must be integers.") from ve
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return fig


def figure_to_image(figure: Figure) -> np.ndarray:
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.

    The supplied figure is closed and inaccessible after this call.
    Args:
        figure: Figure supplied for plotting

    Returns:
        numpy array with the figure's pixels in it's values
    """
    canvas = figure.canvas
    canvas.draw()
    pil_image = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    plt.close()
    return np.array(pil_image)


def show_ndarray_in_matplotlib(img: np.ndarray):
    img = Image.fromarray(img, "RGB")
    img.save("my.png")
    img.show()


def plot_examples_from_dataset(ds: Dataset, idxs: List[int], title: str = "", fig: Figure = None) -> Optional[Figure]:
    """Plot images from the dataset according to $idxs.

    Args:
        ds: torch image Dataset
        idxs: Indices that we want to plot
        title: Title for the plot
        fig: An optional figure to plot on

    Returns:
        Figure is a figure was passed, else it plots the images as described.
    """
    image_datas = [ds[i][0].numpy().swapaxes(0, 2).swapaxes(0, 1) for i in idxs]
    if image_datas[0].dtype == np.float32:
        for i, img in enumerate(image_datas):
            im_max, im_min = max(1.0, img.max()), min(0.0, img.min())
            img = (img - im_min) / (im_max - im_min)
            image_datas[i] = img_as_ubyte(img)
    image_labels = [ds.classes[ds[i][1]] for i in idxs]

    fig_avail = True
    if fig is None:
        fig_avail = False
        fig = plt.figure()
    axes: List[List[Axes]] = fig.subplots(
        len(idxs) // 3 + (1 if len(idxs) % 3 != 0 else 0), min(len(idxs), 3), squeeze=False
    )

    fig.suptitle(title)

    if len(idxs) == 1:
        axes[0][0].imshow(image_datas[0])
        axes[0][0].axis("off")
        (_width, height) = fig.get_size_inches()
        fig.set_size_inches(height, height)

    for j, idx in enumerate(idxs):
        axes[j // 3][j % 3].imshow(image_datas[j])
        axes[j // 3][j % 3].axis("off")
        axes[j // 3][j % 3].set_title(f"{idx} - {image_labels[j]}")
    fig.tight_layout()
    plt.tight_layout()
    if fig_avail:
        return fig
    else:
        plt.show()
        return None
    # TODO: add functionality from matplotlib to bokeh with image in between
