"""Camera stream producer thread."""
import cv2
from PySide6.QtCore import QThread, Signal
from src.config import DEFAULT_CAMERA


class CameraStream(QThread):
    frame_ready = Signal(object)
    error = Signal(str)

    def __init__(self, source=DEFAULT_CAMERA, parent=None):
        super().__init__(parent)
        self.source = source
        self._running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
            self.error.emit(f"Cannot open camera source: {self.source}")
            return
        self._running = True
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.error.emit("Failed to read frame from camera.")
                break
            self.frame_ready.emit(frame)
        self.cleanup()

    def stop(self):
        self._running = False
        self.wait(3000)

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

