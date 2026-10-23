"""Main window UI."""
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QGroupBox, QGridLayout,
                             QComboBox)
from PySide6.QtCore import Qt

from src.config import (WINDOW_WIDTH, WINDOW_HEIGHT,
                        DEFAULT_CAMERA, MODEL_MODES, DEFAULT_MODEL_MODE,
                        LANGUAGES, DEFAULT_LANGUAGE, tr)
from src.threads.camera import CameraStream
from src.threads.inference import InferenceWorker
from src.ui.styles import DARK_STYLE
from src.utils import cv_to_pixmap


def _load_saved_language():
    """Return the persisted language code, or the default."""
    from PySide6.QtCore import QSettings
    lang = QSettings("PPE Safety Monitor", "app").value(
        "language", DEFAULT_LANGUAGE)
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE


class MainWindow(QMainWindow):
    def __init__(self, model_mode=DEFAULT_MODEL_MODE):
        super().__init__()
        self.model_mode = (model_mode if model_mode in MODEL_MODES
                           else DEFAULT_MODEL_MODE)
        self.lang = _load_saved_language()
        self.setWindowTitle(tr(self.lang, "title"))
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(DARK_STYLE)

        self.camera = None
        self.inference = None
        self.running = False
        self._last_stats = {"persons": 0, "safe": 0, "unsafe": 0,
                            "missing_helmet": 0, "missing_vest": 0}

        self._build_ui()
        self._retranslate()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Video area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setObjectName("videoFrame")
        self.video_label.setStyleSheet("background-color:#11111b;border:1px solid #45475a;border-radius:8px;")
        layout.addWidget(self.video_label, stretch=3)

        # Side panel
        side = QVBoxLayout()
        layout.addLayout(side, stretch=1)

        self.ctrl_group = QGroupBox()
        cl = QVBoxLayout(self.ctrl_group)

        mode_row = QHBoxLayout()
        self.lbl_model = QLabel()
        mode_row.addWidget(self.lbl_model)
        self.mode_combo = QComboBox()
        for mode in MODEL_MODES:  # "pt", "onnx"
            self.mode_combo.addItem(mode.upper(), mode)
        idx = self.mode_combo.findData(self.model_mode)
        self.mode_combo.setCurrentIndex(max(idx, 0))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, stretch=1)
        cl.addLayout(mode_row)

        lang_row = QHBoxLayout()
        self.lbl_lang = QLabel()
        lang_row.addWidget(self.lbl_lang)
        self.lang_combo = QComboBox()
        for code, native_name in LANGUAGES.items():
            self.lang_combo.addItem(native_name, code)
        lang_idx = self.lang_combo.findData(self.lang)
        self.lang_combo.setCurrentIndex(max(lang_idx, 0))
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(self.lang_combo, stretch=1)
        cl.addLayout(lang_row)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton()
        self.start_btn.setObjectName("startBtn")
        self.stop_btn = QPushButton()
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        cl.addLayout(btn_row)
        side.addWidget(self.ctrl_group)

        self.stats_group = QGroupBox()
        gl = QGridLayout(self.stats_group)
        self.lbl_persons = QLabel()
        self.lbl_safe = QLabel()
        self.lbl_unsafe = QLabel()
        self.lbl_nh = QLabel()
        self.lbl_nv = QLabel()
        for i, w in enumerate([self.lbl_persons, self.lbl_safe, self.lbl_unsafe,
                                self.lbl_nh, self.lbl_nv]):
            gl.addWidget(w, i, 0)
        side.addWidget(self.stats_group)

        self.status = QLabel()
        side.addWidget(self.status)
        side.addStretch()

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------
    def _on_language_changed(self, index):
        code = self.lang_combo.itemData(index)
        if code is None or code == self.lang:
            return
        self.change_lang(code)

    def change_lang(self, code):
        """Switch the UI language live and persist the choice."""
        if code not in LANGUAGES:
            return
        self.lang = code
        self._retranslate()
        # Persist so the selection survives restarts.
        from PySide6.QtCore import QSettings
        QSettings("PPE Safety Monitor", "app").setValue("language", code)

    def _t(self, key, **kwargs):
        return tr(self.lang, key, **kwargs)

    def _retranslate(self):
        """Re-apply every translated string to the widgets."""
        self.setWindowTitle(self._t("title"))
        self.ctrl_group.setTitle(self._t("controls"))
        self.stats_group.setTitle(self._t("statistics"))
        self.lbl_model.setText(self._t("model_weights"))
        self.lbl_lang.setText(self._t("language"))
        self.start_btn.setText(self._t("start"))
        self.stop_btn.setText(self._t("stop"))
        if not self.running:
            self.video_label.setText(self._t("camera_not_started"))
            self.status.setText(self._t("idle"))
        self._update_stats_labels()
        # Keep the combo in sync when the language is set programmatically.
        idx = self.lang_combo.findData(self.lang)
        if idx >= 0 and self.lang_combo.currentIndex() != idx:
            self.lang_combo.blockSignals(True)
            self.lang_combo.setCurrentIndex(idx)
            self.lang_combo.blockSignals(False)

    def _update_stats_labels(self):
        s = self._last_stats
        self.lbl_persons.setText(f"{self._t('persons')}: {s['persons']}")
        self.lbl_safe.setText(f"{self._t('safe')}: {s['safe']}")
        self.lbl_unsafe.setText(f"{self._t('unsafe')}: {s['unsafe']}")
        self.lbl_nh.setText(f"{self._t('missing_helmet')}: {s['missing_helmet']}")
        self.lbl_nv.setText(f"{self._t('missing_vest')}: {s['missing_vest']}")

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
            self.status.setText(self._t("model_mode", mode=mode.upper()))

    def start(self):
        if self.running:
            return
        self.camera = CameraStream(DEFAULT_CAMERA)
        self.inference = InferenceWorker(mode=self.model_mode, lang=self.lang)
        self.camera.frame_ready.connect(self.on_frame)
        self.inference.result_ready.connect(self.on_result)
        self.camera.error.connect(self.on_error)
        self.inference.error.connect(self.on_error)
        self.camera.start()
        self.inference.start()
        self.running = True
        self.video_label.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.status.setText(self._t("running", mode=self.model_mode.upper()))

    def on_frame(self, frame):
        if self.inference is not None:
            self.inference.submit_frame(frame)

    def on_result(self, frame, stats):
        pix = cv_to_pixmap(frame)
        if pix is not None:
            self.video_label.setPixmap(pix.scaled(self.video_label.size(),
                                                Qt.KeepAspectRatio))
        self._last_stats = stats
        self._update_stats_labels()

    def on_error(self, msg):
        self.status.setText(self._t("error", msg=msg))

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
        self.lang_combo.setEnabled(True)
        self.video_label.setText(self._t("camera_not_started"))
        self.status.setText(self._t("stopped"))

    def closeEvent(self, event):
        self.stop()
        event.accept()
