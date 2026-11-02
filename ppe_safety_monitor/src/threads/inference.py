"""Inference worker consumer thread."""
import logging
import os
import threading
import time

from PySide6.QtCore import QThread, Signal

from src.config import (DEFAULT_MODEL_MODE, MODEL_MODES, DEVICE,
                        CONF_THRESHOLD, IOU_THRESHOLD,
                        OVERLAP_RATIO, CLASS_PERSON, CLASS_NONE,
                        PPE_CLASSES, NO_PPE_MAP, REQUIRED_PPE,
                        COLOR_SAFE, COLOR_DANGER, DEFAULT_LANGUAGE, tr,
                        MIN_GEAR_BOX_SIZE,
                        STAT_KEY_PERSONS, STAT_KEY_SAFE, STAT_KEY_UNSAFE,
                        STAT_KEY_MISSING_HELMET, STAT_KEY_MISSING_VEST,
                        STAT_KEY_MISSING_GLOVES, STAT_KEY_MISSING_GOGGLES,
                        STAT_KEY_MISSING_BOOTS)
from src.utils import (overlap_ratio, check_regional_overlap, get_zone_box,
                       draw_box_unicode, normalize_class_name)

logger = logging.getLogger(__name__)

VALID_MODEL_MODES = tuple(MODEL_MODES.keys())

# Stat dict key for each monitored PPE item ("helmet" -> "missing_helmet").
STAT_KEYS_BY_ITEM = {
    "helmet":  STAT_KEY_MISSING_HELMET,
    "vest":    STAT_KEY_MISSING_VEST,
    "gloves":  STAT_KEY_MISSING_GLOVES,
    "goggles": STAT_KEY_MISSING_GOGGLES,
    "boots":   STAT_KEY_MISSING_BOOTS,
}

# Overlay translation keys per gear item; fall back to a generic label so a
# newly added class never crashes the drawing code.
_OV_WORN_KEYS = {"helmet": "ov_helmet", "vest": "ov_vest",
                 "gloves": "ov_gloves", "boots": "ov_boots",
                 "goggles": "ov_goggles"}
_OV_MISSING_KEYS = {"helmet": "ov_no_helmet_box", "vest": "ov_no_vest_box",
                    "gloves": "ov_no_gloves_box", "boots": "ov_no_boots_box",
                    "goggles": "ov_no_goggles_box"}
_OV_MISSING_SHORT_KEYS = {"helmet": "ov_no_helmet", "vest": "ov_no_vest",
                          "gloves": "ov_no_gloves", "boots": "ov_no_boots",
                          "goggles": "ov_no_goggles"}


def _ov_worn(lang, item):
    key = _OV_WORN_KEYS.get(item)
    return tr(lang, key) if key else item.capitalize()


def _ov_missing(lang, item):
    key = _OV_MISSING_KEYS.get(item)
    return tr(lang, key) if key else f"No {item.capitalize()}"


def _ov_missing_short(lang, item):
    """Short form used inside the person's violation label."""
    key = _OV_MISSING_SHORT_KEYS.get(item)
    return tr(lang, key) if key else f"no {item}"


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

# Map normalized model class names -> internal gear buckets.
# Built from config.PPE_CLASSES so every supported item (helmet, vest,
# gloves, goggles, boots) is recognised regardless of spelling variants.
_PPE_CLASSES = {normalize_class_name(c): normalize_class_name(c)
                for c in PPE_CLASSES}
# Native "missing gear" classes (no_helmet / no_goggle / no_boots ...),
# mapped to the gear item they confirm as missing.
_NO_PPE_CLASSES = {normalize_class_name(k): normalize_class_name(v)
                   for k, v in NO_PPE_MAP.items()}
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
                 lang=DEFAULT_LANGUAGE, ppe_to_monitor=None, parent=None):
        super().__init__(parent)
        self.mode = mode if mode in MODEL_MODES else DEFAULT_MODEL_MODE
        self.model_path = resolve_model_path(mode=self.mode,
                                             model_path=model_path)
        # UI language used for on-frame overlay labels. Written from the GUI
        # thread and read here; a plain str assignment is atomic under the GIL.
        self.lang = lang
        # Which PPE items to enforce this frame ("helmet", "vest", ...).
        # Defaults to config.REQUIRED_PPE; set from settings.json by the UI.
        # Same GIL-atomic single-assignment reasoning as ``lang`` above.
        if ppe_to_monitor is None:
            ppe_to_monitor = REQUIRED_PPE
        self.ppe_to_monitor = [normalize_class_name(p) for p in ppe_to_monitor]
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

    def set_ppe_to_monitor(self, items):
        """Update the enforced PPE list live (GIL-atomic rebinding)."""
        self.ppe_to_monitor = [normalize_class_name(i) for i in items]

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
        stats = {STAT_KEY_PERSONS: 0, STAT_KEY_SAFE: 0, STAT_KEY_UNSAFE: 0}
        for key in (STAT_KEY_MISSING_HELMET, STAT_KEY_MISSING_VEST,
                    STAT_KEY_MISSING_GLOVES, STAT_KEY_MISSING_GOGGLES,
                    STAT_KEY_MISSING_BOOTS):
            stats[key] = 0
        return stats

    @staticmethod
    def _box_too_small(xyxy):
        """Noise filter: YOLO hallucinates tiny boxes for gloves/boots."""
        x1, y1, x2, y2 = xyxy
        return (x2 - x1) < MIN_GEAR_BOX_SIZE or (y2 - y1) < MIN_GEAR_BOX_SIZE

    def _process(self, frame, results):
        stats = self._empty_stats()
        # --- Dynamic settings gating ---------------------------------------
        # Only monitored gear types get pools; unselected classes are parsed
        # but skipped during matching to save CPU/GPU time.
        monitor = [item for item in self.ppe_to_monitor
                   if item in _PPE_CLASSES.values()]
        monitor_set = set(monitor)

        persons = []      # list of (xyxy, conf)
        pools = {item: [] for item in monitor}       # worn-gear candidates
        no_pools = {item: [] for item in monitor}    # native no_* evidence
        none_boxes = []   # explicit "no PPE at all" detections
        unknown_classes = set()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = normalize_class_name(r.names.get(cls_id, str(cls_id)))
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                if name in _PERSON_NAMES:
                    persons.append((xyxy, conf))
                    continue
                item = _PPE_CLASSES.get(name)
                if item is None:
                    item = _NO_PPE_CLASSES.get(name)
                    if item is None:
                        if name == _NONE_CLASS:
                            none_boxes.append((xyxy, conf))
                        else:
                            unknown_classes.add(name)
                        continue
                    # Native missing-gear evidence: only needed when that
                    # item is actually being monitored.
                    if item in monitor_set and not self._box_too_small(xyxy):
                        no_pools[item].append((xyxy, conf))
                    continue
                # Worn-gear candidate: skip entirely if not monitored
                # (dynamic gating) or if it's a sub-threshold noise box.
                if item in monitor_set and not self._box_too_small(xyxy):
                    pools[item].append((xyxy, conf))
        if unknown_classes:
            # Surface typos / unexpected model outputs instead of silently
            # dropping them (logged once per distinct set per frame).
            logger.debug("Ignoring unknown model classes: %s",
                         sorted(unknown_classes))
        if not persons:
            logger.debug("No person detections this frame; expected one of %s",
                         sorted(_PERSON_NAMES))

        # --- Anatomical zonal matching --------------------------------------
        # For each monitored gear type, greedily assign every pool box to at
        # most ONE person: the candidate whose anatomical zone contains the
        # box with the highest containment ratio. Assigned boxes are removed
        # from the pool so they can never be double-counted across two people.
        owned_worn = {item: [[] for _ in persons] for item in monitor}
        for item in monitor:
            remaining = list(pools[item])
            while remaining:
                best = None  # (ratio, person_idx, pool_idx)
                for pi, (pbox, _pc) in enumerate(persons):
                    for gi, (gxy, _gc) in enumerate(remaining):
                        if not check_regional_overlap(pbox, gxy, item):
                            continue
                        ratio = overlap_ratio(get_zone_box(pbox, item), gxy)
                        if best is None or ratio > best[0]:
                            best = (ratio, pi, gi)
                if best is None:
                    break  # no remaining box matches any person's zone
                _ratio, pi, gi = best
                owned_worn[item][pi].append(remaining[gi][0])
                remaining.pop(gi)
            pools[item] = remaining  # leftovers stay unowned (drawn only)

        # Assign native no_* evidence boxes to persons via the same zones.
        owned_missing = {item: [[] for _ in persons] for item in monitor}
        for item in monitor:
            remaining = list(no_pools[item])
            for nxy, _nc in remaining:
                best_idx, best_ratio = -1, 0.0
                for pi, (pbox, _pc) in enumerate(persons):
                    zone = get_zone_box(pbox, item)
                    if not check_regional_overlap(pbox, nxy, item):
                        continue
                    ratio = overlap_ratio(zone, nxy)
                    if ratio > best_ratio:
                        best_idx, best_ratio = pi, ratio
                if best_idx >= 0:
                    owned_missing[item][best_idx].append(nxy)

        none_per_person = [
            any(overlap_ratio(pbox, nxy) >= OVERLAP_RATIO
                for nxy, _c in none_boxes)
            for pbox, _pc in persons
        ]

        # --- Per-person compliance ------------------------------------------
        for idx, (p, _pconf) in enumerate(persons):
            stats[STAT_KEY_PERSONS] += 1
            missing_items = []
            for item in monitor:
                has_worn = bool(owned_worn[item][idx])
                # Secondary confirmation: heuristic says missing, but the
                # model natively detected a no_<item> box on this person.
                has_explicit_missing = bool(owned_missing[item][idx])
                if has_explicit_missing or none_per_person[idx] or not has_worn:
                    missing_items.append(item)

            if missing_items:
                stats[STAT_KEY_UNSAFE] += 1
                color = COLOR_DANGER
                label = (self._t("ov_unsafe") + ": " + ", ".join(
                    self._ov_item_short(m) for m in missing_items))
                for m in missing_items:
                    skey = STAT_KEYS_BY_ITEM.get(m)
                    if skey:
                        stats[skey] += 1
            else:
                stats[STAT_KEY_SAFE] += 1
                color = COLOR_SAFE
                label = self._t("ov_safe")
            draw_box_unicode(frame, p,
                             f"{self._t('ov_person')}: {label}", color)

        # Draw gear boxes (unowned/background boxes are still shown, but only
        # owner-assigned ones influenced the statistics above).
        for item, boxes in pools.items():
            for xyxy, _c in boxes:
                draw_box_unicode(frame, xyxy,
                                 self._ov_worn(item), COLOR_SAFE, 1)
        for item, boxes in no_pools.items():
            for xyxy, _c in boxes:
                draw_box_unicode(frame, xyxy,
                                 self._ov_missing(item), COLOR_DANGER, 1)

        return frame, stats

    def _ov_worn(self, item):
        return _ov_worn(self.lang, item)

    def _ov_missing(self, item):
        return _ov_missing(self.lang, item)

    def _ov_item_short(self, item):
        return _ov_missing_short(self.lang, item)

    def _t(self, key, **kwargs):
        """Translate an overlay label for the current UI language."""
        return tr(self.lang, key, **kwargs)
