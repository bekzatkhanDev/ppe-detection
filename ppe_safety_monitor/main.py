"""Entry point for PPE Safety Monitor."""
import argparse
import sys
from PySide6.QtWidgets import QApplication
from src.config import DEFAULT_MODEL_MODE, MODEL_MODES
from src.ui.main_window import MainWindow


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="PPE Safety Monitor - real-time helmet/vest detection")
    parser.add_argument(
        "--mode", choices=sorted(MODEL_MODES), default=DEFAULT_MODEL_MODE,
        help="Model weights mode: 'pt' (PyTorch) or 'onnx' (ONNX Runtime). "
             "Default: %(default)s")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    app = QApplication(sys.argv)
    win = MainWindow(model_mode=args.mode)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
