"""Login dialog shown before the main window.

Validates credentials against the local SQLite user store and exposes the
authenticated identity via ``logged_in_user`` -> dict(username, role).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFormLayout,
                               QMessageBox)

from src.auth import db_manager as db


class LoginDialog(QDialog):
    def __init__(self, parent=None, warn_default_password=False):
        super().__init__(parent)
        self.logged_in_user = None  # {'username':..., 'role':...} on success
        self.setWindowTitle("PPE Safety Monitor - Login")
        self.setFixedSize(380, 240)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        root = QVBoxLayout(self)
        title = QLabel("🔐 Sign in to continue")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:16px;font-weight:bold;color:#89b4fa;")
        root.addWidget(title)

        form = QFormLayout()
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("Username")
        self.user_edit.setMinimumHeight(30)
        self.pw_edit = QLineEdit()
        self.pw_edit.setPlaceholderText("Password")
        self.pw_edit.setEchoMode(QLineEdit.Password)
        self.pw_edit.setMinimumHeight(30)
        form.addRow("Username:", self.user_edit)
        form.addRow("Password:", self.pw_edit)
        root.addLayout(form)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color:#f38ba8;")
        root.addWidget(self.error_label)

        btns = QHBoxLayout()
        self.login_btn = QPushButton("Login")
        self.login_btn.setObjectName("startBtn")
        self.login_btn.setMinimumHeight(34)
        cancel = QPushButton("Cancel")
        cancel.setMinimumHeight(34)
        cancel.clicked.connect(self.reject)
        self.login_btn.clicked.connect(self.try_login)
        btns.addWidget(self.login_btn)
        btns.addWidget(cancel)
        root.addLayout(btns)

        self.pw_edit.returnPressed.connect(self.try_login)
        self.user_edit.returnPressed.connect(self.try_login)
        self.user_edit.setFocus()

        # Inline warning (non-modal) so it is visible & testable.
        self.warn_label = QLabel("")
        self.warn_label.setAlignment(Qt.AlignCenter)
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color:#f9e2af;")
        root.addWidget(self.warn_label)
        if warn_default_password:
            self.warn_label.setText(
                "Default Admin account created on first run. "
                "Change the password after logging in.")

    def try_login(self):
        username = self.user_edit.text().strip()
        password = self.pw_edit.text()
        if not username or not password:
            self._fail("Please enter both username and password.")
            return
        ok, role = db.verify_user(username, password)
        if ok:
            self.logged_in_user = {"username": username, "role": role}
            # Convenience hint for the freshly seeded default account.
            if (username == db.DEFAULT_ADMIN_USERNAME
                    and password == db.DEFAULT_ADMIN_PASSWORD):
                self.warn_label.setText(
                    "Signed in with the DEFAULT admin password - "
                    "change it via Manage Users.")
                # Give the label a moment to paint before closing.
                self.warn_label.repaint()
            self.accept()
        else:
            # Generic message: never reveal whether the user exists.
            self._fail("Invalid username or password.")

    def _fail(self, msg):
        self.error_label.setText(msg)
        self.pw_edit.selectAll()
        self.pw_edit.setFocus()
