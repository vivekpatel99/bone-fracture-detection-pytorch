import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def unnormalize_bbox(image, coords):
    """
    Unnormalizes bounding box coordinates from [0, 1] to pixels.

    Parameters
    ----------
    image : ndarray
        The image the bounding box is drawn on.
    coords : iterable
        The normalized coordinates of the bounding box.

    Returns
    -------
    coords : ndarray
        The unnormalized coordinates of the bounding box.
    """

    h, w, _ = image.shape
    coords = np.array(coords).reshape(-1, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    return coords


def plot_bbox(image:np.ndarray, class_id:int, label:str,  bbox:np.ndarray) -> None:
    """
    Plots a bounding box onto an image.

    Parameters
    ----------
    image : ndarray
        The image to plot the bounding box on.
    class_id : int
        The class ID of the object in the bounding box.
    bbox : iterable
        The coordinates of the bounding box, normalized to [0, 1].

    Returns
    -------
    None
    """

    coords = unnormalize_bbox(image, bbox)
    coords = coords.astype(int)
    # for i in range(len(coords)):
    #     cv2.line(image, tuple(coords[i]), tuple(
    #         coords[(i + 1) % len(coords)]), (0, 255, 0), 2)
    xmin = coords[:, 0].min()
    ymin = coords[:, 1].min()
    xmax = coords[:, 0].max()
    ymax = coords[:, 1].max()
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.putText(image, str(class_id), tuple(
        coords[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    plt.title(label)
    plt.imshow(image)
    plt.show()




def plot_multiple_images(image_paths:list[str], bboxes:list[np.ndarray], class_ids:int, class_mapping:dict) -> None:

    """
    Plots multiple images with bounding boxes.

    Parameters
    ----------
    images : list
        A list of paths to images.
    bboxes : list
        A list of bounding boxes, where each bounding box is an iterable of
        normalized coordinates [x1, y1, x2, y2].
    class_ids : list
        A list of class IDs, where each class ID is an integer.
    class_mapping : dict
        A dictionary mapping class IDs to class labels.

    Returns
    -------
    None
    """
    def _plot_bbox(image,  bbox):
        coords = unnormalize_bbox(image, bbox)
        coords = coords.astype(int)
        # for i in range(len(coords)):
        #     cv2.line(image, tuple(coords[i]), tuple(
        #         coords[(i + 1) % len(coords)]), (0, 255, 0), 2)
        xmin = coords[:, 0].min()
        ymin = coords[:, 1].min()
        xmax = coords[:, 0].max()
        ymax = coords[:, 1].max()
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        return image
    random_idx = random.sample(range(len(image_paths)), 8)
    print(random_idx)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, ax in zip(random_idx, axes):
        image = cv2.imread(image_paths[idx])
        bbox = bboxes[idx]
        class_label = class_mapping[class_ids[idx]]
        image_with_bbox = _plot_bbox(image, bbox)
        ax.imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
        ax.set_title(class_label)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
