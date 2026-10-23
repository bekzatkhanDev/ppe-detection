"""Utility helpers for PPE Safety Monitor."""
import cv2
from PySide6.QtGui import QImage, QPixmap


def normalize_class_name(name):
    """Normalize a model class name for robust matching.

    Lower-cases and treats '_' and '-' as equivalent so that e.g.
    'Person', 'no-helmet' and 'no_helmet' all match their config entries.
    """
    return str(name).strip().lower().replace("-", "_")


def iou(box_a, box_b):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def overlap_ratio(box_a, box_b):
    """Fraction of box_b's area that lies inside box_a (containment ratio).

    PPE detections (helmet, vest) are much smaller than the person box that
    encloses them, so symmetric IoU is far too strict for "is this item worn
    by this person?". The containment ratio answers exactly that: how much of
    the PPE box falls within the person box.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    if area_b <= 0:
        return 0.0
    return inter_area / area_b


def cv_to_pixmap(frame):
    """Convert an OpenCV BGR frame to QPixmap (deep copy, safe across threads)."""
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    # copy() detaches the QImage from the temporary numpy buffer so the
    # pixmap stays valid after `rgb` is garbage collected.
    return QPixmap.fromImage(qimg.copy())


def draw_box(frame, box, label, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
