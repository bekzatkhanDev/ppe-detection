"""Global configuration for PPE Safety Monitor."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model weight modes.
# The app supports two interchangeable backends via Ultralytics YOLO:
#   * PyTorch mode (.pt)  - full ultralytics + torch stack, most accurate/flexible
#   * ONNX mode (.onnx)   - lightweight export, faster CPU inference, no torch
# Select the active mode here (or from the UI / with `python main.py --mode onnx`).
MODEL_MODES = {
    "pt": os.path.join(MODELS_DIR, "best.pt"),
    "onnx": os.path.join(MODELS_DIR, "best.onnx"),
}
DEFAULT_MODEL_MODE = "pt"
# Kept for backwards compatibility; overridden by the selected mode at runtime.
MODEL_PATH = MODEL_MODES[DEFAULT_MODEL_MODE]

# Device selection for inference: None = auto (CUDA if available else CPU),
# or force e.g. "cpu" / "cuda:0". Applies to both pt and onnx modes.
DEVICE = None

# Camera sources: 0 for webcam, or path/URL for video files/streams
DEFAULT_CAMERA = 0

# Detection classes expected from the trained YOLO model.
# The bundled model (models/best.pt / best.onnx) was trained with these names:
# {0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none,
#  6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots}
# Matching is case-insensitive and treats '_' and '-' as equivalent.
CLASS_PERSON = "person"
CLASS_HELMET = "helmet"
CLASS_VEST = "vest"
CLASS_NO_HELMET = "no_helmet"
CLASS_NO_VEST = "no_vest"

# PPE items that must be worn for a person to be considered safe.
REQUIRED_PPE = [CLASS_HELMET, CLASS_VEST]

# Colors (BGR for OpenCV drawing)
COLOR_SAFE = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_PERSON = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)

# Thresholds
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
OVERLAP_RATIO = 0.5  # fraction of the PPE box inside the person box to count as "worn"

# UI
WINDOW_TITLE = "PPE Safety Monitor"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_TARGET = 30



LANGUAGES = ["en", "ru", "kk"]
TRANSLATIONS = {
    "en": {"title": "PPE Safety Monitor", "start": "Start Monitoring", "stop": "Stop",
           "persons": "Persons", "safe": "Safe", "unsafe": "Unsafe",
           "missing_helmet": "Missing Helmet", "missing_vest": "Missing Vest",
           "idle": "Idle", "running": "Running", "stopped": "Stopped"},
    "ru": {"title": "Мониторинг СИЗ", "start": "Запустить", "stop": "Остановить",
           "persons": "Люди", "safe": "Безопасно", "unsafe": "Опасно",
           "missing_helmet": "Нет каски", "missing_vest": "Нет жилета",
           "idle": "Ожидание", "running": "Работает", "stopped": "Остановлено"},
    "kk": {"title": "ЖКҚ бақылау", "start": "Бастау", "stop": "Тоқтату",
           "persons": "Адамдар", "safe": "Қауіпсіз", "unsafe": "Қауіпті",
           "missing_helmet": "Каска жоқ", "missing_vest": "Жилет жоқ",
           "idle": "Күту", "running": "Жұмыс", "stopped": "Тоқтатылды"},
}
