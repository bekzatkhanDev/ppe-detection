"""Inference worker consumer thread."""
import logging
import os
import threading
import time

from PySide6.QtCore import QThread, Signal

from src.config import (DEFAULT_MODEL_MODE, MODEL_MODES, DEVICE,
                        CONF_THRESHOLD, IOU_THRESHOLD,
                        OVERLAP_RATIO, CLASS_PERSON, CLASS_HELMET, CLASS_VEST,
                        CLASS_NO_HELMET, CLASS_NO_VEST, CLASS_NONE,
                        REQUIRED_PPE,
                        COLOR_SAFE, COLOR_DANGER, DEFAULT_LANGUAGE, tr,
                        STAT_KEY_PERSONS, STAT_KEY_SAFE, STAT_KEY_UNSAFE,
                        STAT_KEY_MISSING_HELMET, STAT_KEY_MISSING_VEST)
from src.utils import overlap_ratio, draw_box_unicode, normalize_class_name

logger = logging.getLogger(__name__)

VALID_MODEL_MODES = tuple(MODEL_MODES.keys())


def resolve_model_path(mode=None, model_path=None):
    """Resolve the weights file to use.

    An explicit ``model_path`` always wins; otherwise the ``mode`` key
    ("pt" or "onnx") selects from config.MODEL_MODES. Falls back to the
    default mode for unknown values.

    Precedence note: a custom ``model_path`` passed to the constructor is
    discarded if :meth:`InferenceWorker.set_mode` is called afterwards --
    switching modes intentionally resets to the configured weights.
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
# Explicit "no PPE at all" class (bundled model's class 5). A person whose
# box contains this detection is missing *every* required item.
_NONE_CLASS = normalize_class_name(CLASS_NONE)

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

    # Throttle repeated error emissions so a persistent failure does not
    # spam the UI / signal chain at video frame rate.
    _ERROR_EMIT_INTERVAL = 5.0  # seconds between identical messages

    def __init__(self, model_path=None, mode=DEFAULT_MODEL_MODE,
                 lang=DEFAULT_LANGUAGE, parent=None):
        super().__init__(parent)
        self.mode = mode if mode in MODEL_MODES else DEFAULT_MODEL_MODE
        self.model_path = resolve_model_path(mode=self.mode,
                                             model_path=model_path)
        # UI language used for on-frame overlay labels. Written from the GUI
        # thread and read here; a plain str assignment is atomic under the GIL.
        self.lang = lang
        self.model = None
        self._running = False
        self._latest_frame = None
        # Real mutex protecting the shared frame slot between the GUI thread
        # (submit_frame) and this worker thread (run).
        self._lock_frame = threading.Lock()
        self._last_error_msg = None
        self._last_error_time = 0.0

    def set_mode(self, mode):
        """Select the weights mode ('pt' or 'onnx'). Call before start()."""
        if mode not in MODEL_MODES:
            raise ValueError(
                f"Unknown model mode {mode!r}; expected one of "
                f"{VALID_MODEL_MODES}")
        if self._running:
            raise RuntimeError("Cannot change model mode while running")
        self.mode = mode
        # Note: switching modes deliberately overrides any custom model_path
        # supplied to the constructor (see resolve_model_path docstring).
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

    def _emit_error(self, msg):
        """Emit an error, throttling repeats of the same message."""
        now = time.monotonic()
        if (msg == self._last_error_msg
                and now - self._last_error_time < self._ERROR_EMIT_INTERVAL):
            return
        self._last_error_msg = msg
        self._last_error_time = now
        logger.error("Inference error: %s", msg)
        if self._running:
            self.error.emit(msg)

    def run(self):
        if not self.load_model():
            return
        self._running = True
        while self._running:
            frame = self._take_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            # Separate inference failures from annotation failures: a
            # cosmetic drawing bug must not be reported as a model error,
            # and we still show the raw frame instead of dropping it.
            try:
                predict_kwargs = dict(conf=CONF_THRESHOLD,
                                      iou=IOU_THRESHOLD, verbose=False)
                if DEVICE:
                    predict_kwargs["device"] = DEVICE
                results = self.model.predict(frame, **predict_kwargs)
            except Exception as e:
                self._emit_error(f"Inference failed: {e}")
                continue
            try:
                annotated, stats = self._process(frame, results)
            except Exception as e:
                logger.exception("Annotation failed")
                self._emit_error(f"Annotation failed: {e}")
                annotated, stats = frame, self._empty_stats()
            if self._running:
                self.result_ready.emit(annotated, stats)

    def stop(self):
        self._running = False
        # Wait long enough that even a slow inference pass finishes before
        # the GUI tears down; emitting into a destroyed UI would crash.
        self.wait(30000)

    @staticmethod
    def _empty_stats():
        return {STAT_KEY_PERSONS: 0, STAT_KEY_SAFE: 0, STAT_KEY_UNSAFE: 0,
                STAT_KEY_MISSING_HELMET: 0, STAT_KEY_MISSING_VEST: 0}

    def _process(self, frame, results):
        stats = self._empty_stats()
        persons = []      # list of (xyxy, conf)
        ppe_worn = {}     # item -> list of (xyxy, conf)
        ppe_missing = {}  # item -> list of (xyxy, conf)
        none_boxes = []   # explicit "no PPE at all" detections
        items = set(_PPE_CLASSES.values()) | set(_NO_PPE_CLASSES.values())
        for item in items:
            ppe_worn[item] = []
            ppe_missing[item] = []

        unknown_classes = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = normalize_class_name(r.names.get(cls_id, str(cls_id)))
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                if name in _PERSON_NAMES:
                    persons.append((xyxy, conf))
                elif name in _PPE_CLASSES:
                    ppe_worn[_PPE_CLASSES[name]].append((xyxy, conf))
                elif name in _NO_PPE_CLASSES:
                    ppe_missing[_NO_PPE_CLASSES[name]].append((xyxy, conf))
                elif name == _NONE_CLASS:
                    none_boxes.append((xyxy, conf))
                else:
                    unknown_classes.add(name)
        if unknown_classes:
            # Surface typos / unexpected model outputs instead of silently
            # dropping them (logged once per distinct set per frame).
            logger.debug("Ignoring unknown model classes: %s",
                         sorted(unknown_classes))
        if not persons:
            logger.debug("No person detections this frame; expected one of %s",
                         sorted(_PERSON_NAMES))

        # --- Assign each PPE box to exactly one owner -----------------------
        # A PPE box belongs to the person with the highest containment ratio.
        # Boxes whose best match falls below OVERLAP_RATIO are treated as
        # background/unowned: they are drawn but ignored for safety stats.
        # This prevents one helmet/vest box from being counted for several
        # overlapping people.
        owned_worn = {i: [[] for _ in persons] for i in ppe_worn}
        owned_missing = {i: [[] for _ in persons] for i in ppe_missing}
        for target, source in ((owned_worn, ppe_worn),
                               (owned_missing, ppe_missing)):
            for item, boxes in source.items():
                for xyxy, _conf in boxes:
                    best_idx, best_ratio = -1, 0.0
                    for idx, (pbox, _pc) in enumerate(persons):
                        # PPE boxes are much smaller than person boxes, so
                        # measure how much of the PPE box lies inside the
                        # person box (containment ratio) rather than
                        # symmetric IoU.
                        ratio = overlap_ratio(pbox, xyxy)
                        if ratio > best_ratio:
                            best_idx, best_ratio = idx, ratio
                    if best_idx >= 0 and best_ratio >= OVERLAP_RATIO:
                        target[item][best_idx].append(xyxy)

        none_per_person = [
            any(overlap_ratio(pbox, nxy) >= OVERLAP_RATIO
                for nxy, _c in none_boxes)
            for pbox, _pc in persons
        ]

        for idx, (p, _pconf) in enumerate(persons):
            stats[STAT_KEY_PERSONS] += 1
            missing_items = []
            for item in REQUIRED_PPE:
                key = _PPE_CLASSES.get(normalize_class_name(item))
                if key is None:
                    continue
                has_worn = bool(owned_worn[key][idx])
                has_explicit_missing = bool(owned_missing[key][idx])
                # Unsafe if an explicit no-PPE box overlaps the person, if
                # the model says they wear nothing at all ("none" class), or
                # if no matching PPE box overlaps them.
                if has_explicit_missing or none_per_person[idx] or not has_worn:
                    missing_items.append(key)

            if missing_items:
                stats[STAT_KEY_UNSAFE] += 1
                color = COLOR_DANGER
                label = (self._t("ov_unsafe") + ": " + ", ".join(
                    self._t("ov_no_helmet") if m == "helmet"
                    else self._t("ov_no_vest") for m in missing_items))
                for m in missing_items:
                    if m == "helmet":
                        stats[STAT_KEY_MISSING_HELMET] += 1
                    elif m == "vest":
                        stats[STAT_KEY_MISSING_VEST] += 1
            else:
                stats[STAT_KEY_SAFE] += 1
                color = COLOR_SAFE
                label = self._t("ov_safe")
            draw_box_unicode(frame, p,
                             f"{self._t('ov_person')}: {label}", color)

        # Draw PPE boxes (unowned/background boxes are still shown, but only
        # owner-assigned ones influenced the statistics above).
        for item, boxes in ppe_worn.items():
            for xyxy, _c in boxes:
                draw_box_unicode(frame, xyxy,
                                 self._t("ov_helmet") if item == "helmet"
                                 else self._t("ov_vest"), COLOR_SAFE, 1)
        for item, boxes in ppe_missing.items():
            for xyxy, _c in boxes:
                draw_box_unicode(frame, xyxy,
                                 self._t("ov_no_helmet_box")
                                 if item == "helmet"
                                 else self._t("ov_no_vest_box"),
                                 COLOR_DANGER, 1)

        return frame, stats

    def _t(self, key, **kwargs):
        """Translate an overlay label for the current UI language."""
        return tr(self.lang, key, **kwargs)
