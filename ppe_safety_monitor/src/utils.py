"""Utility helpers for PPE Safety Monitor."""
import cv2
import numpy as np
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


# --- Unicode-aware overlay text ---------------------------------------------
# cv2.putText only renders ASCII; Cyrillic / Kazakh labels would show up as
# "????" on the video overlay. For non-ASCII strings we rasterise the text
# with Qt (which handles full Unicode) and blit it onto the frame instead.

_QT_FONT = None
_QT_FONT_PIXEL_SIZE = 18


def _qt_font():
    global _QT_FONT
    if _QT_FONT is None:
        from PySide6.QtGui import QFont
        _QT_FONT = QFont("Segoe UI", -1, QFont.Weight.Normal)
        _QT_FONT.setPixelSize(_QT_FONT_PIXEL_SIZE)
    return _QT_FONT


def _render_text_qt(text):
    """Render ``text`` to an RGBA numpy array using Qt (full Unicode)."""
    from PySide6.QtCore import QRectF, Qt as QtCore
    from PySide6.QtGui import QImage, QPainter, QColor

    font = _qt_font()
    # Measure with a throwaway image first.
    probe = QImage(1, 1, QImage.Format_ARGB32)
    p = QPainter(probe)
    p.setFont(font)
    try:
        bounds = p.boundingRect(QRectF(0, 0, 10000, 1000), int(QtCore.TextSingleLine), text)
    finally:
        p.end()
    w = int(bounds.width()) + 8
    h = int(bounds.height()) + 6
    if w <= 0 or h <= 0:
        return None

    img = QImage(w, h, QImage.Format_RGBA8888)
    img.fill(0)
    p = QPainter(img)
    try:
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255, 255))
        p.drawText(img.rect(), int(QtCore.AlignVCenter | QtCore.AlignLeft), text)
    finally:
        p.end()

    # QImage rows are 4-byte aligned; bytesPerLine accounts for any padding.
    bpl = img.bytesPerLine()
    buf = img.constBits()
    try:
        buf.setsize(bpl * h)  # PySide6 < 6.7 returns a raw void pointer
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)
    except AttributeError:  # newer PySide6 returns a memoryview already sized
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)
    arr = raw[:, :w * 4].reshape(h, w, 4).copy()
    return arr


def put_text_unicode(frame, text, org, color_bgr=(255, 255, 255)):
    """Draw ``text`` at ``org`` (baseline-style top-left anchor).

    Falls back to cv2.putText for pure-ASCII strings (fast path); uses a
    Qt-rendered bitmap for anything containing non-ASCII characters so that
    Russian / Kazakh overlay labels render correctly.
    """
    try:
        text.encode("ascii")
        is_ascii = True
    except UnicodeEncodeError:
        is_ascii = False

    if is_ascii:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color_bgr, 2)
        return frame

    arr = _render_text_qt(text)
    if arr is None:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color_bgr, 2)
        return frame

    h, w = arr.shape[:2]
    x, y = int(org[0]), int(org[1])
    fh, fw = frame.shape[:2]
    x2, y2 = min(x + w, fw), min(y + h, fh)
    if x2 <= x or y2 <= y:
        return frame
    arr = arr[:y2 - y, :x2 - x]
    region = frame[y:y2, x:x2]
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    bgr = arr[:, :, 2::-1].astype(np.float32)  # RGBA -> BGR
    blended = region.astype(np.float32) * (1.0 - alpha) + bgr * alpha
    np.clip(blended, 0, 255, out=blended)
    region[:] = blended.astype(np.uint8)
    return frame


def draw_label_box(frame, x1, y1, x2, label, color):
    """Draw a filled label chip above a box and write the label inside it.

    Unicode-safe counterpart of the label portion of :func:`draw_box`.
    """
    try:
        label.encode("ascii")
        is_ascii = True
    except UnicodeEncodeError:
        is_ascii = False

    if is_ascii:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    else:
        arr = _render_text_qt(label)
        if arr is not None:
            th, tw = arr.shape[:2]
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            is_ascii = True

    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    if is_ascii:
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        put_text_unicode(frame, label, (x1 + 2, y1 - th - 4))
    return frame


def draw_box_unicode(frame, box, label, color, thickness=2):
    """``draw_box`` variant that supports non-ASCII (translated) labels."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label_box(frame, x1, y1, x2, label, color)
    return frame
