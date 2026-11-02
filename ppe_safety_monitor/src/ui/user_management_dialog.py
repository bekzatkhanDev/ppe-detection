"""Admin-only user management dialog.

Lists all users in a QTableWidget and lets an admin add or delete accounts.
Guards enforced:
  * cannot delete the account you are currently logged in with;
  * cannot delete the last remaining admin.
Optionally, admins can change a selected user's password via the
"Change Password..." button (uses db.set_password).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QComboBox,
                               QTableWidget, QTableWidgetItem,
                               QAbstractItemView, QMessageBox,
                               QGroupBox, QFormLayout, QHeaderView,
                               QInputDialog)

from src.auth import db_manager as db


class UserManagementDialog(QDialog):
    def __init__(self, current_username, parent=None):
        super().__init__(parent)
        self.current_username = current_username
        self.setWindowTitle("Manage Users")
        self.resize(560, 480)

        root = QVBoxLayout(self)

        # ---- Users table -------------------------------------------------
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "Username", "Role"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, stretch=1)

        del_row = QHBoxLayout()
        self.delete_btn = QPushButton("🗑 Delete Selected User")
        self.delete_btn.setObjectName("stopBtn")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.pw_btn = QPushButton("🔑 Change Password...")
        self.pw_btn.clicked.connect(self.change_password)
        del_row.addWidget(self.delete_btn)
        del_row.addWidget(self.pw_btn)
        root.addLayout(del_row)

        # ---- Add-user form ----------------------------------------------
        add_group = QGroupBox("➕ Add User")
        form = QFormLayout(add_group)
        self.new_user = QLineEdit()
        self.new_user.setPlaceholderText("Username")
        self.new_pw = QLineEdit()
        self.new_pw.setPlaceholderText("Password")
        self.new_pw.setEchoMode(QLineEdit.Password)
        self.role_combo = QComboBox()
        self.role_combo.addItem("User", db.ROLE_USER)
        self.role_combo.addItem("Admin", db.ROLE_ADMIN)
        form.addRow("Username:", self.new_user)
        form.addRow("Password:", self.new_pw)
        form.addRow("Role:", self.role_combo)

        btns = QHBoxLayout()
        btns.addStretch()
        self.add_btn = QPushButton("Add User")
        self.add_btn.setObjectName("startBtn")
        self.add_btn.clicked.connect(self.add_user)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(self.add_btn)
        btns.addWidget(close_btn)
        form.addRow(btns)
        root.addWidget(add_group)

        self.refresh_table()

    # ------------------------------------------------------------------
    def refresh_table(self):
        users = db.get_all_users()
        self.table.setRowCount(len(users))
        for r, u in enumerate(users):
            id_item = QTableWidgetItem(str(u["id"]))
            name_item = QTableWidgetItem(u["username"])
            role_txt = "Admin" if u["role"] == db.ROLE_ADMIN else "User"
            role_item = QTableWidgetItem(role_txt)
            for item in (id_item, name_item, role_item):
                item.setTextAlignment(Qt.AlignCenter)
            if u["username"] == self.current_username:
                name_item.setText(f"{u['username']}  (you)")
            if u["role"] == db.ROLE_ADMIN:
                role_item.setForeground(_brush("#f9e2af"))
            self.table.setItem(r, 0, id_item)
            self.table.setItem(r, 1, name_item)
            self.table.setItem(r, 2, role_item)
            # stash raw username for deletion lookup
            name_item.setData(Qt.UserRole, u["username"])
            role_item.setData(Qt.UserRole, u["role"])

    def _selected(self):
        row = self.table.currentRow()
        if row < 0:
            return None, None
        username = self.table.item(row, 1).data(Qt.UserRole)
        role = self.table.item(row, 2).data(Qt.UserRole)
        return username, role

    # ------------------------------------------------------------------
    def add_user(self):
        username = self.new_user.text().strip()
        password = self.new_pw.text()
        role = self.role_combo.currentData()
        if not username or not password:
            QMessageBox.warning(self, "Validation",
                                "Username and password are required.")
            return
        try:
            created = db.create_user(username, password, role)
        except ValueError as exc:
            QMessageBox.warning(self, "Validation", str(exc))
            return
        if not created:
            QMessageBox.warning(self, "Duplicate",
                                f"Username '{username}' already exists.")
            return
        self.new_user.clear()
        self.new_pw.clear()
        self.refresh_table()

    def delete_selected(self):
        username, role = self._selected()
        if username is None:
            QMessageBox.information(self, "No selection",
                                    "Select a user row to delete.")
            return
        # Guard 1: never delete yourself.
        if username == self.current_username:
            QMessageBox.critical(
                self, "Not allowed",
                "You cannot delete the account you are logged in with.")
            return
        # Guard 2: never remove the last admin.
        if role == db.ROLE_ADMIN and db.count_admins() <= 1:
            QMessageBox.critical(
                self, "Not allowed",
                "This is the last remaining Admin account. Create another "
                "admin before deleting this one.")
            return
        answer = QMessageBox.question(
            self, "Confirm delete",
            f"Delete user '{username}'? This cannot be undone.")
        if answer != QMessageBox.Yes:
            return
        db.delete_user(username)
        self.refresh_table()

    def change_password(self):
        username, _ = self._selected()
        if username is None:
            QMessageBox.information(self, "No selection",
                                    "Select a user first.")
            return
        new_pw, ok = QInputDialog.getText(
            self, "Change password",
            f"New password for '{username}':", QLineEdit.Password)
        if not ok:
            return
        if not new_pw:
            QMessageBox.warning(self, "Validation",
                                "Password cannot be empty.")
            return
        db.set_password(username, new_pw)
        QMessageBox.information(self, "Done",
                                f"Password updated for '{username}'.")


def _brush(hex_color):
    from PySide6.QtGui import QColor, QBrush
    return QBrush(QColor(hex_color))
