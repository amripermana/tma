import numpy as np
import cv2
import math
from shapely.geometry import Polygon


def regularize_rboxes_numpy(rboxes):
    """
    Regularize rotated boxes from range [0, pi] into [0, pi/2].

    Args:
        rboxes (np.ndarray): Array of shape (N, 5) with [cx, cy, w, h, theta(rad)].

    Returns:
        np.ndarray: Regularized boxes in same format with theta in [0, pi/2].
    """
    x, y, w, h, t = np.split(rboxes, 5, axis=-1)

    # Kondisi: jika theta >= pi/2 maka swap w dan h
    swap = (t % np.pi) >= (np.pi / 2)
    
    w_, h_ = np.where(swap, h, w), np.where(swap, w, h)
    
    # Normalisasi theta ke [0, pi/2]
    t = t % (np.pi / 2)

    return np.concatenate([x, y, w_, h_, t], axis=-1)


def xywhr2xyxyxyxy(input):
    """
    Convert OBB from [cx, cy, w, h, angle] to 4 corner points [pt1, pt2, pt3, pt4].
    """
    x = regularize_rboxes_numpy(input)

    cos, sin, cat, stack = np.cos, np.sin, np.concatenate, np.stack

    ctr = x[..., :2]  # center (cx, cy)
    w, h, angle = (x[..., i:i + 1] for i in range(2, 5))

    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, axis=-1)
    vec2 = cat(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], axis=-2)


def iou(box1, box2):
    # IOU untuk bbox biasa [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (area1 + area2 - inter_area + 1e-6)

def polygon_iou(poly1, poly2):
    """Compute IoU between two polygons."""
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0

def obb_nms(corners, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on rotated bounding boxes.

    Args:
        corners (ndarray): Array of shape (N, 4, 2) representing box corners.
        scores (ndarray): Array of shape (N,) with confidence scores.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        List of indices of boxes to keep.
    """
    idxs = np.argsort(-scores)
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        ious = np.array([
            polygon_iou(corners[current], corners[i]) for i in idxs[1:]
        ])

        idxs = idxs[1:][ious < iou_threshold]

    return keep




