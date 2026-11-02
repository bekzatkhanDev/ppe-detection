"""Camera stream producer thread."""
import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal

from src.config import DEFAULT_CAMERA, FPS_TARGET


class CameraStream(QThread):
    frame_ready = Signal(object)
    error = Signal(str)

    def __init__(self, source=DEFAULT_CAMERA, parent=None):
        super().__init__(parent)
        self.source = source
        self._running = False
        self.cap = None

    def _open_capture(self):
        """Try platform-friendly backends first, then the default backend."""
        backends = []
        if sys.platform.startswith("win"):
            # DirectShow is faster/more reliable on Windows webcams.
            backends.append(cv2.CAP_DSHOW)
        elif sys.platform == "darwin":
            backends.append(cv2.CAP_AVFOUNDATION)
        else:
            backends.append(cv2.CAP_V4L2)
        backends.append(cv2.CAP_ANY)  # let OpenCV auto-select

        for backend in backends:
            cap = cv2.VideoCapture(self.source, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def run(self):
        self.cap = self._open_capture()
        if self.cap is None:
            self.error.emit(f"Cannot open camera source: {self.source}")
            return

        self._running = True
        is_video_file = isinstance(self.source, str)
        frame_interval = 1.0 / max(FPS_TARGET, 1)
        reopen_failures = 0

        while self._running:
            start = time.monotonic()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if is_video_file:
                    # Loop video files instead of ending the stream. If even
                    # the seek/reopen fails, back off instead of hot-spinning.
                    if self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        continue
                    reopen_failures += 1
                    if reopen_failures > 3:
                        self.error.emit(
                            f"Cannot restart video source: {self.source}")
                        break
                    cap = self._open_capture()
                    if cap is None:
                        time.sleep(0.5)
                        continue
                    self.cap.release()
                    self.cap = cap
                    continue
                self.error.emit("Failed to read frame from camera.")
                break
            reopen_failures = 0
            self.frame_ready.emit(frame)
            # Pace capture towards FPS_TARGET to avoid unbounded queueing.
            elapsed = time.monotonic() - start
            sleep_left = frame_interval - elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)

        self.cleanup()

    def stop(self):
        self._running = False
        self.wait(3000)

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None