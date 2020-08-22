from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.figure import Figure
from skimage import img_as_ubyte
from torch.utils.data import Dataset


def plot_confusion_matrix(confusion_matrix, class_names, figsize=(10, 8), fontsize=14):
    """Plots a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly, constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the outputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure.
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig


def figure_to_image(figure) -> np.ndarray:
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
      returns it. The supplied figure is closed and inaccessible after this call."""
    canvas = figure.canvas
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(),
                                canvas.tostring_rgb())
    plt.close()
    return np.array(pil_image)


def show_ndarray_in_matplotlib(img: np.ndarray):
    img = Image.fromarray(img, 'RGB')
    img.save('my.png')
    img.show()


def plot_examples_from_dataset(ds: Dataset, idxs: List[int], title=""):
    """
    Plot images from the dataset according to $idxs
    Args:
        ds: torch image Dataset
        idxs: Indices that we want to plot
        title: Title for the plot
    """
    image_datas = [ds[i][0].numpy().swapaxes(0, 2).swapaxes(0, 1) for i in idxs]
    if image_datas[0].dtype == np.float32:
        image_datas = [img_as_ubyte(img) for img in image_datas]
    image_labels = [ds.classes[ds[i][1]] for i in idxs]

    if len(idxs) == 1:
        plt.imshow(image_datas[0])
        plt.title(title)
    else:
        f, axarr = plt.subplots(1, len(idxs))
        f.suptitle(title)
        for j, idx in enumerate(idxs):
            axarr[j].imshow(image_datas[j])
            axarr[j].set_title(f"{idx} - {image_labels[j]}")
    plt.show()
