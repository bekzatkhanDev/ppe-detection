"""Dark mode QSS styling."""
DARK_STYLE = """
QMainWindow { background-color: #1e1e2e; color: #cdd6f4; }
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: Segoe UI, sans-serif; font-size: 13px; }
QLabel { color: #cdd6f4; }
QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { background-color: #181825; color: #6c7086; }
QComboBox, QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #89b4fa; }
QFrame#videoFrame { background-color: #11111b; border: 1px solid #45475a; border-radius: 8px; }
"""
