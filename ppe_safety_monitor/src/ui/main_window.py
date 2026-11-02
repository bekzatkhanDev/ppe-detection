"""Main window UI."""
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QGroupBox, QGridLayout,
                             QComboBox)
from PySide6.QtCore import Qt

from src.config import (WINDOW_WIDTH, WINDOW_HEIGHT,
                        DEFAULT_CAMERA, MODEL_MODES, DEFAULT_MODEL_MODE,
                        LANGUAGES, DEFAULT_LANGUAGE, tr, load_settings,
                        save_settings, PPE_CLASSES,
                        STAT_KEY_PERSONS, STAT_KEY_SAFE, STAT_KEY_UNSAFE,
                        STAT_KEY_MISSING_HELMET, STAT_KEY_MISSING_VEST,
                        STAT_KEY_MISSING_GLOVES, STAT_KEY_MISSING_GOGGLES,
                        STAT_KEY_MISSING_BOOTS)
from src.threads.camera import CameraStream
from src.threads.inference import InferenceWorker
from src.ui.settings_dialog import SettingsDialog
from src.ui.user_management_dialog import UserManagementDialog
from src.ui.styles import DARK_STYLE
from src.utils import cv_to_pixmap


def _load_saved_language():
    """Return the persisted language code, or the default."""
    from PySide6.QtCore import QSettings
    lang = QSettings("PPE Safety Monitor", "app").value(
        "language", DEFAULT_LANGUAGE)
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE


class MainWindow(QMainWindow):
    def __init__(self, model_mode=DEFAULT_MODEL_MODE,
                 username="guest", role="user"):
        super().__init__()
        self.model_mode = (model_mode if model_mode in MODEL_MODES
                           else DEFAULT_MODEL_MODE)
        # Authenticated identity (passed in from the login flow).
        self.username = username or "guest"
        self.role = role if role in ("admin", "user") else "user"
        self.is_admin = self.role == "admin"
        self.lang = _load_saved_language()
        self.setWindowTitle(tr(self.lang, "title"))
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(DARK_STYLE)

        # Active monitoring configuration, loaded from settings.json
        # (falls back to defaults: camera 0, helmet + vest).
        settings = load_settings()
        self.camera_sources = list(settings["camera_sources"])
        self.ppe_to_monitor = list(settings["ppe_to_monitor"])

        self.cameras = []       # list of CameraStream
        self.inferences = []    # list of InferenceWorker
        self.running = False
        self._restarting = False
        self._last_stats = {STAT_KEY_PERSONS: 0, STAT_KEY_SAFE: 0,
                            STAT_KEY_UNSAFE: 0,
                            STAT_KEY_MISSING_HELMET: 0,
                            STAT_KEY_MISSING_VEST: 0,
                            STAT_KEY_MISSING_GLOVES: 0,
                            STAT_KEY_MISSING_GOGGLES: 0,
                            STAT_KEY_MISSING_BOOTS: 0}

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

        # Side panel (20% right column)
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

        # ⚙️ Settings button: opens the camera / PPE selection dialog.
        self.settings_btn = QPushButton()
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.clicked.connect(self.open_settings)
        cl.addWidget(self.settings_btn)

        # 👥 Manage Users button: admin-only (hidden for standard users).
        self.users_btn = QPushButton()
        self.users_btn.setObjectName("usersBtn")
        self.users_btn.clicked.connect(self.open_user_management)
        cl.addWidget(self.users_btn)

        # Current monitoring configuration summary.
        self.cfg_label = QLabel()
        self.cfg_label.setWordWrap(True)
        self.cfg_label.setStyleSheet("color:#89b4fa;")
        cl.addWidget(self.cfg_label)

        side.addWidget(self.ctrl_group)

        self.stats_group = QGroupBox()
        gl = QGridLayout(self.stats_group)
        self.lbl_persons = QLabel()
        self.lbl_safe = QLabel()
        self.lbl_unsafe = QLabel()
        self.lbl_nh = QLabel()
        self.lbl_nv = QLabel()
        self.lbl_ng = QLabel()
        self.lbl_ngo = QLabel()
        self.lbl_nb = QLabel()
        self._stat_labels = [self.lbl_persons, self.lbl_safe, self.lbl_unsafe,
                             self.lbl_nh, self.lbl_nv, self.lbl_ng,
                             self.lbl_ngo, self.lbl_nb]
        for i, w in enumerate(self._stat_labels):
            gl.addWidget(w, i, 0)
        side.addWidget(self.stats_group)

        self.status = QLabel()
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        side.addStretch()

        # "Logged in as" label pinned to the bottom corner of the panel.
        self.user_label = QLabel()
        self.user_label.setObjectName("userLabel")
        self.user_label.setStyleSheet(
            "color:#a6adc8;font-size:12px;padding:4px;")
        self.user_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        side.addWidget(self.user_label)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self._update_cfg_label()
        self._apply_role_access()

    # ------------------------------------------------------------------
    # Role-based access control
    # ------------------------------------------------------------------
    def _apply_role_access(self):
        """Show admin-only controls and the identity label."""
        role_name = "Admin" if self.is_admin else "User"
        self.user_label.setText(
            f"Logged in as: {self.username} ({role_name})")
        self.settings_btn.setVisible(self.is_admin)
        self.users_btn.setVisible(self.is_admin)

    def open_user_management(self):
        """Admin-only: open the user management dialog."""
        if not self.is_admin:  # programmatic guard, on top of hidden button
            return
        dlg = UserManagementDialog(self.username, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Settings dialog & apply flow
    # ------------------------------------------------------------------
    def open_settings(self):
        """Show the Settings dialog; restart monitoring on Save & Apply."""
        if not self.is_admin:  # programmatic guard, on top of hidden button
            return
        dlg = SettingsDialog(parent=self)
        dlg.settings_applied.connect(self.apply_settings)
        dlg.exec()

    def apply_settings(self, camera_sources, ppe_to_monitor):
        """Persist new settings and safely restart threads with them."""
        if not camera_sources:
            camera_sources = [DEFAULT_CAMERA]
        self.camera_sources = list(camera_sources)
        self.ppe_to_monitor = list(ppe_to_monitor)
        try:
            save_settings({"camera_sources": self.camera_sources,
                           "ppe_to_monitor": self.ppe_to_monitor})
        except OSError as exc:
            self.on_error(f"Could not save settings: {exc}")
        self._update_cfg_label()
        if self.running:
            # Stop current camera/inference threads, then relaunch with the
            # newly selected sources and gear. Guard against re-entrancy.
            if self._restarting:
                return
            self._restarting = True
            try:
                self.stop()
                self.start()
            finally:
                self._restarting = False
        self.status.setText(self._t("settings_applied"))

    def _update_cfg_label(self):
        cams = ", ".join(str(c) for c in self.camera_sources)
        gear = ", ".join(p.capitalize() for p in self.ppe_to_monitor) or "-"
        self.cfg_label.setText(f"Cameras: {cams}\nPPE: {gear}")

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
        self.settings_btn.setText(self._t("settings"))
        role_name = "Admin" if self.is_admin else "User"
        self.user_label.setText(
            f"{self._t('logged_in_as')}: {self.username} ({role_name})")
        self.users_btn.setText(self._t("manage_users"))
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
        self.lbl_persons.setText(
            f"{self._t('persons')}: {s[STAT_KEY_PERSONS]}")
        self.lbl_safe.setText(f"{self._t('safe')}: {s[STAT_KEY_SAFE]}")
        self.lbl_unsafe.setText(f"{self._t('unsafe')}: {s[STAT_KEY_UNSAFE]}")
        self.lbl_nh.setText(
            f"{self._t('missing_helmet')}: {s[STAT_KEY_MISSING_HELMET]}")
        self.lbl_nv.setText(
            f"{self._t('missing_vest')}: {s[STAT_KEY_MISSING_VEST]}")
        self.lbl_ng.setText(
            f"{self._t('missing_gloves')}: {s.get(STAT_KEY_MISSING_GLOVES, 0)}")
        self.lbl_ngo.setText(
            f"{self._t('missing_goggles')}: {s.get(STAT_KEY_MISSING_GOGGLES, 0)}")
        self.lbl_nb.setText(
            f"{self._t('missing_boots')}: {s.get(STAT_KEY_MISSING_BOOTS, 0)}")
        # Only show stat rows for gear actually being monitored.
        missing_keys_by_item = {
            "helmet": STAT_KEY_MISSING_HELMET, "vest": STAT_KEY_MISSING_VEST,
            "gloves": STAT_KEY_MISSING_GLOVES,
            "goggles": STAT_KEY_MISSING_GOGGLES,
            "boots": STAT_KEY_MISSING_BOOTS}
        label_by_key = {
            STAT_KEY_MISSING_HELMET: self.lbl_nh,
            STAT_KEY_MISSING_VEST: self.lbl_nv,
            STAT_KEY_MISSING_GLOVES: self.lbl_ng,
            STAT_KEY_MISSING_GOGGLES: self.lbl_ngo,
            STAT_KEY_MISSING_BOOTS: self.lbl_nb}
        for item in PPE_CLASSES:
            key = missing_keys_by_item[item]
            label_by_key[key].setVisible(item in self.ppe_to_monitor)

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
        self.cameras = []
        self.inferences = []
        for src in self.camera_sources:
            camera = CameraStream(src)
            inference = InferenceWorker(mode=self.model_mode, lang=self.lang,
                                        ppe_to_monitor=self.ppe_to_monitor)
            camera.frame_ready.connect(
                lambda frame, inf=inference: inf.submit_frame(frame))
            inference.result_ready.connect(self.on_result)
            camera.error.connect(self.on_error)
            inference.error.connect(self.on_error)
            camera.start()
            inference.start()
            self.cameras.append(camera)
            self.inferences.append(inference)
        self.running = True
        self.video_label.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.status.setText(self._t("running", mode=self.model_mode.upper()))

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
        # Stop the consumers first so they never receive frames from a
        # camera that is about to be torn down. stop() waits for each thread.
        for inference in self.inferences:
            inference.stop()
        self.inferences = []
        for camera in self.cameras:
            camera.stop()
        self.cameras = []
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
