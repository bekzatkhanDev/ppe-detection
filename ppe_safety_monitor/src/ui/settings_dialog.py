"""Settings dialog: camera auto-detection/selection and PPE monitoring choices."""
import cv2
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QLabel, QPushButton, QCheckBox)

from src.config import PPE_CLASSES, load_settings


def scan_available_cameras(max_index=4):
    """Probe cv2.VideoCapture(i) for indices 0..max_index.

    Returns the list of integer indices that open successfully and can grab
    at least one frame. Each capture is released immediately so cameras are
    free for the monitoring threads afterwards.
    """
    available = []
    for idx in range(0, max_index + 1):
        cap = None
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _frame = cap.read()
                if ok:
                    available.append(idx)
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return available


class SettingsDialog(QDialog):
    """Modal settings editor. Emits ``settings_applied`` on Save & Apply."""

    settings_applied = Signal(list, list)  # camera_sources, ppe_to_monitor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        settings = load_settings()
        selected_cams = set(settings["camera_sources"])
        selected_ppe = set(settings["ppe_to_monitor"])

        # --- Cameras -------------------------------------------------------
        cam_group = QGroupBox("Cameras")
        cam_layout = QVBoxLayout(cam_group)
        status = QLabel()
        cam_layout.addWidget(status)
        self.camera_checks = {}
        found = scan_available_cameras(4)  # indices 0..4
        if found:
            status.setText(f"{len(found)} camera(s) detected.")
            for idx in found:
                cb = QCheckBox(f"Camera {idx}")
                cb.setChecked(idx in selected_cams)
                cam_layout.addWidget(cb)
                self.camera_checks[idx] = cb
        else:
            status.setText("No cameras detected (indices 0-4). "
                           "Connect a camera or edit settings.json manually.")
            # Keep previously saved selections editable even when no camera
            # currently opens (e.g. headless/dev machine).
            for idx in sorted(selected_cams):
                cb = QCheckBox(f"Camera {idx} (offline)")
                cb.setChecked(True)
                cam_layout.addWidget(cb)
                self.camera_checks[idx] = cb
        layout.addWidget(cam_group)

        # --- PPE gear ------------------------------------------------------
        ppe_group = QGroupBox("PPE to monitor")
        ppe_layout = QVBoxLayout(ppe_group)
        ppe_layout.addWidget(QLabel(
            "Violations are checked only for the selected gear "
            "(vest relies on the absence heuristic)."))
        self.ppe_checks = {}
        for item in PPE_CLASSES:
            cb = QCheckBox(item.capitalize())
            cb.setChecked(item in selected_ppe)
            ppe_layout.addWidget(cb)
            self.ppe_checks[item] = cb
        layout.addWidget(ppe_group)

        # --- Buttons ---------------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save && Apply")
        self.save_btn.setObjectName("startBtn")
        self.save_btn.setDefault(True)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self._on_save)

    def _on_save(self):
        cams = [idx for idx, cb in self.camera_checks.items() if cb.isChecked()]
        ppe = [item for item, cb in self.ppe_checks.items() if cb.isChecked()]
        self.settings_applied.emit(sorted(cams), ppe)
        self.accept()
