"""Entry point for PPE Safety Monitor."""
import argparse
import sys
from PySide6.QtWidgets import QApplication
from src.auth import db_manager
from src.config import DEFAULT_MODEL_MODE, MODEL_MODES
from src.ui.login_dialog import LoginDialog
from src.ui.main_window import MainWindow


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="PPE Safety Monitor - real-time helmet/vest detection")
    parser.add_argument(
        "--mode", choices=sorted(MODEL_MODES), default=DEFAULT_MODEL_MODE,
        help="Model weights mode: 'pt' (PyTorch) or 'onnx' (ONNX Runtime). "
             "Default: %(default)s")
    parser.add_argument(
        "--no-login", action="store_true",
        help="Skip the login dialog (guest/user role). Useful for testing.")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    app = QApplication(sys.argv)

    username, role = "guest", "user"
    if not args.no_login:
        # Create/seed the users table; warn if the default admin was made.
        first_run = db_manager.init_db()
        dlg = LoginDialog(warn_default_password=first_run)
        if dlg.exec() != LoginDialog.Accepted:
            sys.exit(0)  # user cancelled -> quit silently
        user = dlg.logged_in_user or {}
        username, role = user.get("username", "guest"), user.get("role", "user")

    win = MainWindow(model_mode=args.mode, username=username, role=role)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
