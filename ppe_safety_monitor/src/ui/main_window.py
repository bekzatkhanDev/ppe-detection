"""Main window UI."""
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QGroupBox, QGridLayout,
                             QComboBox)
from PySide6.QtCore import Qt

from src.config import (WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
                        DEFAULT_CAMERA, MODEL_MODES, DEFAULT_MODEL_MODE)
from src.threads.camera import CameraStream
from src.threads.inference import InferenceWorker
from src.ui.styles import DARK_STYLE
from src.utils import cv_to_pixmap


class MainWindow(QMainWindow):
    def __init__(self, model_mode=DEFAULT_MODEL_MODE):
        super().__init__()
        self.model_mode = (model_mode if model_mode in MODEL_MODES
                           else DEFAULT_MODEL_MODE)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(DARK_STYLE)

        self.camera = None
        self.inference = None
        self.running = False

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Video area
        self.video_label = QLabel("Camera not started")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setObjectName("videoFrame")
        self.video_label.setStyleSheet("background-color:#11111b;border:1px solid #45475a;border-radius:8px;")
        layout.addWidget(self.video_label, stretch=3)

        # Side panel
        side = QVBoxLayout()
        layout.addLayout(side, stretch=1)

        ctrl = QGroupBox("Controls")
        cl = QVBoxLayout(ctrl)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Model weights:"))
        self.mode_combo = QComboBox()
        for mode in MODEL_MODES:  # "pt", "onnx"
            self.mode_combo.addItem(mode.upper(), mode)
        idx = self.mode_combo.findData(self.model_mode)
        self.mode_combo.setCurrentIndex(max(idx, 0))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, stretch=1)
        cl.addLayout(mode_row)

        self.start_btn = QPushButton("Start Monitoring")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        cl.addWidget(self.start_btn)
        cl.addWidget(self.stop_btn)
        side.addWidget(ctrl)

        stats = QGroupBox("Statistics")
        gl = QGridLayout(stats)
        self.lbl_persons = QLabel("Persons: 0")
        self.lbl_safe = QLabel("Safe: 0")
        self.lbl_unsafe = QLabel("Unsafe: 0")
        self.lbl_nh = QLabel("Missing Helmet: 0")
        self.lbl_nv = QLabel("Missing Vest: 0")
        for i, w in enumerate([self.lbl_persons, self.lbl_safe, self.lbl_unsafe,
                                self.lbl_nh, self.lbl_nv]):
            gl.addWidget(w, i, 0)
        side.addWidget(stats)

        self.status = QLabel("Idle")
        side.addWidget(self.status)
        side.addStretch()

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def _on_mode_changed(self, index):
        mode = self.mode_combo.itemData(index)
        if mode is None or mode == self.model_mode:
            return
        if self.running:
            # Restart monitoring so the new weights backend takes effect.
            self.stop()
            self.model_mode = mode
            self.start()
        else:
            self.model_mode = mode
            self.status.setText(f"Model mode: {mode.upper()}")

    def start(self):
        if self.running:
            return
        self.camera = CameraStream(DEFAULT_CAMERA)
        self.inference = InferenceWorker(mode=self.model_mode)
        self.camera.frame_ready.connect(self.on_frame)
        self.inference.result_ready.connect(self.on_result)
        self.camera.error.connect(self.on_error)
        self.inference.error.connect(self.on_error)
        self.camera.start()
        self.inference.start()
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.status.setText(f"Running ({self.model_mode.upper()} weights)")

    def on_frame(self, frame):
        if self.inference is not None:
            self.inference.submit_frame(frame)

    def on_result(self, frame, stats):
        pix = cv_to_pixmap(frame)
        if pix is not None:
            self.video_label.setPixmap(pix.scaled(self.video_label.size(),
                                                Qt.KeepAspectRatio))
        self.lbl_persons.setText(f"Persons: {stats['persons']}")
        self.lbl_safe.setText(f"Safe: {stats['safe']}")
        self.lbl_unsafe.setText(f"Unsafe: {stats['unsafe']}")
        self.lbl_nh.setText(f"Missing Helmet: {stats['missing_helmet']}")
        self.lbl_nv.setText(f"Missing Vest: {stats['missing_vest']}")

    def on_error(self, msg):
        self.status.setText(f"Error: {msg}")

    def stop(self):
        if not self.running:
            return
        # Stop the consumer first so it never receives frames from a camera
        # that is about to be torn down. stop() already waits for the thread.
        if self.inference:
            self.inference.stop()
            self.inference = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.status.setText("Stopped")

    def closeEvent(self, event):
        self.stop()
        event.accept()
