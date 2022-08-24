import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import io
from PIL import Image


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # https://stackoverflow.com/a/61754995
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_rgb(image, min_val=None, max_val=None):
    """Function to normalize 3 band input image to RGB 0-255 image.

    Args:
        image (array_like): Image array to convert to RGB image with dtype
                uint8 [bands, height, width].
        min_val (array_like): Pixel values in
        image less than or equal to this are set to zero in the RGB output.
        max_val (array_like): Pixel values in image greater than
            or equal to this are set to zero in the RGB output.

    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    if image.shape[0] != 3:
        raise ValueError("Must be 3 channel in dimension 1 of image. Found {image.shape[0]}")
    if min_val is None:
        min_val = image.min(axis=-1).min(axis=-1)
    if max_val is None:
        max_val = image.max(axis=-1).max(axis=-1)
    new_image = np.transpose(image, axes=(1, 2, 0))
    new_image = (new_image - min_val) / (max_val - min_val) * 255
    new_image[new_image < 0] = 0
    new_image[new_image > 255] = 255
    return new_image.astype(np.uint8)


def plot_with_bboxes(image, img_bands=None, truth_boxes=None, predicted_boxes=None, predicted_scores=None):
    """Function to plot image together with ground truth boxes, predicted boxes, and predicted scores.

    Args:
        image (np.array): Image array to visualize [bands, height, width].
        img_bands (List[int, int, int]): bands to be mapped to R, G, B
        truth_boxes (array_like): array of ground truth boxes of shape [N, 4]
        predicted_boxes (array_like): array of predicted boxes of shape [N, 4]
        predicted_scores (array_like): array of predicted scores of shape [N,]

    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    assert type(image) == np.ndarray
    fig, ax = plt.subplots(figsize=(6, 6))
    if img_bands is None:
        img_bands = [0, 2, 3]
    rgb_img = get_rgb(image[img_bands])
    ax.imshow(rgb_img)

    # visualize ground truth boxes
    if truth_boxes is not None:
        for box in truth_boxes:
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none'))

    # visualize prediction boxes
    if predicted_boxes is not None:
        for ii in range(len(predicted_boxes)):
            x1, y1, x2, y2 = predicted_boxes[ii]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'))
            plt.text(x1, y1,
                     s=f'{predicted_scores[ii]:.3f}',
                     color='white', verticalalignment='top',
                     bbox={'color': 'red', 'pad': 0})
    plt.tight_layout()
    plot = fig2img(fig)
    plt.close(fig)
    return plot


def plot_batch(images, targets, preds, num=4):
    """Wrapper around plot_with_bboxes to plot multiple images"""
    plots = []
    for i in range(num):
        image = images[i].detach().numpy()
        bands = [0, 2, 3]
        truth_boxes = targets[i]['boxes'].detach().numpy()
        predicted_boxes = preds[i]['boxes'].detach().numpy()
        predicted_scores = preds[i]['scores'].detach().numpy()
        plot = plot_with_bboxes(image, bands, truth_boxes, predicted_boxes, predicted_scores)
        plots.append(np.array(plot))
    return np.hstack(plots)
