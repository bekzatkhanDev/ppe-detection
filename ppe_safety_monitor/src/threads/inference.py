"""Inference worker consumer thread."""
import os
import threading
import time

from PySide6.QtCore import QThread, Signal

from src.config import (DEFAULT_MODEL_MODE, MODEL_MODES, DEVICE,
                        CONF_THRESHOLD, IOU_THRESHOLD,
                        OVERLAP_RATIO, CLASS_PERSON, CLASS_HELMET, CLASS_VEST,
                        CLASS_NO_HELMET, CLASS_NO_VEST, REQUIRED_PPE,
                        COLOR_SAFE, COLOR_DANGER)
from src.utils import overlap_ratio, draw_box, normalize_class_name

VALID_MODEL_MODES = tuple(MODEL_MODES.keys())


def resolve_model_path(mode=None, model_path=None):
    """Resolve the weights file to use.

    An explicit ``model_path`` always wins; otherwise the ``mode`` key
    ("pt" or "onnx") selects from config.MODEL_MODES. Falls back to the
    default mode for unknown values.
    """
    if model_path:
        return model_path
    if mode not in MODEL_MODES:
        mode = DEFAULT_MODEL_MODE
    return MODEL_MODES[mode]

# Map normalized model class names -> internal buckets.
_PPE_CLASSES = {
    normalize_class_name(CLASS_HELMET): "helmet",
    normalize_class_name(CLASS_VEST): "vest",
}
_NO_PPE_CLASSES = {
    normalize_class_name(CLASS_NO_HELMET): "helmet",
    normalize_class_name(CLASS_NO_VEST): "vest",
}

_PERSON_NAMES = {normalize_class_name(CLASS_PERSON)}
# Fallbacks for models trained with differently-named person classes
# (e.g. the bundled model uses capitalised 'Person').
if normalize_class_name(CLASS_PERSON) != "person":
    _PERSON_NAMES.add("person")
else:
    _PERSON_NAMES.add("people")


class InferenceWorker(QThread):
    result_ready = Signal(object, dict)  # annotated frame, stats
    error = Signal(str)

    def __init__(self, model_path=None, mode=DEFAULT_MODEL_MODE, parent=None):
        super().__init__(parent)
        self.mode = mode if mode in MODEL_MODES else DEFAULT_MODEL_MODE
        self.model_path = resolve_model_path(mode=self.mode,
                                             model_path=model_path)
        self.model = None
        self._running = False
        self._latest_frame = None
        # Real mutex protecting the shared frame slot between the GUI thread
        # (submit_frame) and this worker thread (run).
        self._lock_frame = threading.Lock()

    def set_mode(self, mode):
        """Select the weights mode ('pt' or 'onnx'). Call before start()."""
        if mode not in MODEL_MODES:
            raise ValueError(
                f"Unknown model mode {mode!r}; expected one of "
                f"{VALID_MODEL_MODES}")
        if self._running:
            raise RuntimeError("Cannot change model mode while running")
        self.mode = mode
        self.model_path = resolve_model_path(mode=mode)

    def load_model(self):
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model not found: {self.model_path} "
                                f"(mode: {self.mode})")
                return False
            # Ultralytics dispatches on file extension: '.pt' loads the
            # PyTorch weights, '.onnx' runs the ONNX Runtime backend.
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            self.error.emit(f"Failed to load model ({self.mode}): {e}")
            return False

    def submit_frame(self, frame):
        """Hand off the newest frame (drops older unprocessed frames)."""
        with self._lock_frame:
            self._latest_frame = frame

    def _take_frame(self):
        with self._lock_frame:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def run(self):
        if not self.load_model():
            return
        self._running = True
        while self._running:
            frame = self._take_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                predict_kwargs = dict(conf=CONF_THRESHOLD,
                                      iou=IOU_THRESHOLD, verbose=False)
                if DEVICE:
                    predict_kwargs["device"] = DEVICE
                results = self.model.predict(frame, **predict_kwargs)
                annotated, stats = self._process(frame, results)
                if self._running:
                    self.result_ready.emit(annotated, stats)
            except Exception as e:
                if self._running:
                    self.error.emit(str(e))

    def stop(self):
        self._running = False
        self.wait(3000)

    def _process(self, frame, results):
        stats = {"persons": 0, "safe": 0, "unsafe": 0, "missing_helmet": 0,
                 "missing_vest": 0}
        persons = []
        # Buckets per PPE item: worn boxes and explicit "no-<item>" boxes.
        items = set(_PPE_CLASSES.values()) | set(_NO_PPE_CLASSES.values())
        ppe = {}
        for item in items:
            ppe[item] = []
            ppe[item + "_missing"] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = normalize_class_name(r.names.get(cls_id, str(cls_id)))
                xyxy = box.xyxy[0].tolist()
                if name in _PERSON_NAMES:
                    persons.append(xyxy)
                elif name in _PPE_CLASSES:
                    ppe[_PPE_CLASSES[name]].append(xyxy)
                elif name in _NO_PPE_CLASSES:
                    ppe[_NO_PPE_CLASSES[name] + "_missing"].append(xyxy)

        worn_map = {item: ppe[item] for item in _PPE_CLASSES.values()}
        missing_map = {item: ppe[item + "_missing"]
                       for item in _NO_PPE_CLASSES.values()}

        for p in persons:
            stats["persons"] += 1
            missing_items = []
            for item in REQUIRED_PPE:
                key = _PPE_CLASSES.get(normalize_class_name(item))
                if key is None:
                    continue
                # PPE boxes are much smaller than person boxes, so measure how
                # much of the PPE box lies inside the person box (containment
                # ratio) rather than symmetric IoU.
                has_worn = any(
                    overlap_ratio(p, b) >= OVERLAP_RATIO
                    for b in worn_map[key])
                has_explicit_missing = any(
                    overlap_ratio(p, b) >= OVERLAP_RATIO
                    for b in missing_map[key])
                # Unsafe if an explicit no-PPE box overlaps the person, or if
                # no matching PPE box overlaps them at all.
                if has_explicit_missing or not has_worn:
                    missing_items.append(key)

            if missing_items:
                stats["unsafe"] += 1
                color = COLOR_DANGER
                label = "Unsafe: " + ", ".join(
                    f"no {m}" for m in missing_items)
                for m in missing_items:
                    if m == "helmet":
                        stats["missing_helmet"] += 1
                    elif m == "vest":
                        stats["missing_vest"] += 1
            else:
                stats["safe"] += 1
                color = COLOR_SAFE
                label = "Safe"
            draw_box(frame, p, f"Person: {label}", color)

        # Draw PPE boxes
        for h in ppe["helmet"]:
            draw_box(frame, h, "Helmet", COLOR_SAFE, 1)
        for v in ppe["vest"]:
            draw_box(frame, v, "Vest", COLOR_SAFE, 1)
        for nh in ppe["helmet_missing"]:
            draw_box(frame, nh, "No Helmet", COLOR_DANGER, 1)
        for nv in ppe["vest_missing"]:
            draw_box(frame, nv, "No Vest", COLOR_DANGER, 1)

        return frame, stats
