"""Inference worker consumer thread."""
import os
from PySide6.QtCore import QThread, Signal
from src.config import (MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
                       OVERLAP_IOU, CLASS_PERSON, CLASS_HELMET, CLASS_VEST,
                       CLASS_NO_HELMET, CLASS_NO_VEST, COLOR_SAFE, COLOR_DANGER,
                       COLOR_PERSON, COLOR_TEXT)
from src.utils import iou, draw_box


class InferenceWorker(QThread):
    result_ready = Signal(object, dict)  # annotated frame, stats
    error = Signal(str)

    def __init__(self, model_path=MODEL_PATH, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.model = None
        self._running = False
        self._latest_frame = None
        self._lock_frame = False

    def load_model(self):
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model not found: {self.model_path}")
                return False
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
            return False

    def submit_frame(self, frame):
        self._latest_frame = frame

    def run(self):
        if not self.load_model():
            return
        self._running = True
        import time
        while self._running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            self._latest_frame = None
            try:
                results = self.model.predict(frame, conf=CONF_THRESHOLD,
                                           iou=IOU_THRESHOLD, verbose=False)
                annotated, stats = self._process(frame, results)
                self.result_ready.emit(annotated, stats)
            except Exception as e:
                self.error.emit(str(e))

    def stop(self):
        self._running = False
        self.wait(3000)

    def _process(self, frame, results):
        stats = {"persons": 0, "safe": 0, "unsafe": 0, "missing_helmet": 0,
                 "missing_vest": 0}
        persons = []
        helmets = []
        vests = []
        no_helmets = []
        no_vests = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = r.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                if name == CLASS_PERSON:
                    persons.append(xyxy)
                elif name in (CLASS_HELMET,):
                    helmets.append(xyxy)
                elif name in (CLASS_VEST,):
                    vests.append(xyxy)
                elif name in (CLASS_NO_HELMET,):
                    no_helmets.append(xyxy)
                elif name in (CLASS_NO_VEST,):
                    no_vests.append(xyxy)

        for p in persons:
            stats["persons"] += 1
            has_helmet = any(iou(p, h) >= OVERLAP_IOU for h in helmets) or \
                not any(iou(p, nh) >= OVERLAP_IOU for nh in no_helmets)
            # If explicit no-helmet box overlaps person -> missing
            missing_helmet = any(iou(p, nh) >= OVERLAP_IOU for nh in no_helmets)
            missing_vest = any(iou(p, nv) >= OVERLAP_IOU for nv in no_vests)
            # Also consider absence of helmet/vest boxes
            if not missing_helmet and not any(iou(p, h) >= OVERLAP_IOU for h in helmets) \
                    and (CLASS_NO_HELMET in [CLASS_NO_HELMET]):
                # only flag if model reports no-helmet class present globally; keep conservative
                pass

            safe = not missing_helmet and not missing_vest
            if safe:
                stats["safe"] += 1
                color = COLOR_SAFE
                label = "Safe"
            else:
                stats["unsafe"] += 1
                color = COLOR_DANGER
                label = "Unsafe"
                if missing_helmet:
                    stats["missing_helmet"] += 1
                if missing_vest:
                    stats["missing_vest"] += 1
            draw_box(frame, p, f"Person: {label}", color)

        # Draw PPE boxes
        for h in helmets:
            draw_box(frame, h, "Helmet", COLOR_SAFE, 1)
        for v in vests:
            draw_box(frame, v, "Vest", COLOR_SAFE, 1)
        for nh in no_helmets:
            draw_box(frame, nh, "No Helmet", COLOR_DANGER, 1)
        for nv in no_vests:
            draw_box(frame, nv, "No Vest", COLOR_DANGER, 1)

        return frame, stats
