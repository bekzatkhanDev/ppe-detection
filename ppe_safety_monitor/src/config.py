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
# NOTE: the bundled model has no explicit "no_vest" class; CLASS_NO_VEST is
# kept so models trained with that label are supported. The "none" class
# (a person wearing no PPE at all) is handled via CLASS_NONE below.
CLASS_PERSON = "person"
CLASS_HELMET = "helmet"
CLASS_VEST = "vest"
CLASS_NO_HELMET = "no_helmet"
CLASS_NO_VEST = "no_vest"
# Class used by some models for a bare worker without any protective gear.
CLASS_NONE = "none"

# PPE items that must be worn for a person to be considered safe.
REQUIRED_PPE = [CLASS_HELMET, CLASS_VEST]

# Keys of the per-frame statistics dict produced by the inference worker and
# consumed by the UI. Shared constants avoid implicit string coupling.
STAT_KEY_PERSONS = "persons"
STAT_KEY_SAFE = "safe"
STAT_KEY_UNSAFE = "unsafe"
STAT_KEY_MISSING_HELMET = "missing_helmet"
STAT_KEY_MISSING_VEST = "missing_vest"
STAT_KEYS = (STAT_KEY_PERSONS, STAT_KEY_SAFE, STAT_KEY_UNSAFE,
             STAT_KEY_MISSING_HELMET, STAT_KEY_MISSING_VEST)

# Colors (BGR for OpenCV drawing)
COLOR_SAFE = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_DANGER = (0, 0, 255)
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



# Supported UI languages: code -> native display name.
LANGUAGES = {
    "en": "English",
    "ru": "Русский",
    "kk": "Қазақша",
}
DEFAULT_LANGUAGE = "en"

TRANSLATIONS = {
    "en": {
        "title": "PPE Safety Monitor",
        "controls": "Controls",
        "model_weights": "Model weights:",
        "language": "Language:",
        "statistics": "Statistics",
        "start": "Start Monitoring",
        "stop": "Stop",
        "persons": "Persons",
        "safe": "Safe",
        "unsafe": "Unsafe",
        "missing_helmet": "Missing Helmet",
        "missing_vest": "Missing Vest",
        "camera_not_started": "Camera not started",
        "idle": "Idle",
        "running": "Running ({mode} weights)",
        "stopped": "Stopped",
        "error": "Error: {msg}",
        "model_mode": "Model mode: {mode}",
        # Overlay labels drawn onto video frames (OpenCV).
        "ov_person": "Person",
        "ov_safe": "Safe",
        "ov_unsafe": "Unsafe",
        "ov_no_helmet": "no helmet",
        "ov_no_vest": "no vest",
        "ov_helmet": "Helmet",
        "ov_vest": "Vest",
        "ov_no_helmet_box": "No Helmet",
        "ov_no_vest_box": "No Vest",
    },
    "ru": {
        "title": "Мониторинг СИЗ",
        "controls": "Управление",
        "model_weights": "Веса модели:",
        "language": "Язык:",
        "statistics": "Статистика",
        "start": "Запустить мониторинг",
        "stop": "Остановить",
        "persons": "Люди",
        "safe": "Безопасно",
        "unsafe": "Опасно",
        "missing_helmet": "Нет каски",
        "missing_vest": "Нет жилета",
        "camera_not_started": "Камера не запущена",
        "idle": "Ожидание",
        "running": "Работает ({mode})",
        "stopped": "Остановлено",
        "error": "Ошибка: {msg}",
        "model_mode": "Режим модели: {mode}",
        "ov_person": "Человек",
        "ov_safe": "Безопасно",
        "ov_unsafe": "Опасно",
        "ov_no_helmet": "нет каски",
        "ov_no_vest": "нет жилета",
        "ov_helmet": "Каска",
        "ov_vest": "Жилет",
        "ov_no_helmet_box": "Нет каски",
        "ov_no_vest_box": "Нет жилета",
    },
    "kk": {
        "title": "ЖКҚ бақылауы",
        "controls": "Басқару",
        "model_weights": "Модель салмақтары:",
        "language": "Тіл:",
        "statistics": "Статистика",
        "start": "Бақылауды бастау",
        "stop": "Тоқтату",
        "persons": "Адамдар",
        "safe": "Қауіпсіз",
        "unsafe": "Қауіпті",
        "missing_helmet": "Каска жоқ",
        "missing_vest": "Жилет жоқ",
        "camera_not_started": "Камера іске қосылмаған",
        "idle": "Күту",
        "running": "Жұмыс істеуде ({mode})",
        "stopped": "Тоқтатылды",
        "error": "Қате: {msg}",
        "model_mode": "Модель режимі: {mode}",
        "ov_person": "Адам",
        "ov_safe": "Қауіпсіз",
        "ov_unsafe": "Қауіпті",
        "ov_no_helmet": "каска жоқ",
        "ov_no_vest": "жилет жоқ",
        "ov_helmet": "Каска",
        "ov_vest": "Жилет",
        "ov_no_helmet_box": "Каска жоқ",
        "ov_no_vest_box": "Жилет жоқ",
    },
}


def tr(lang, key, **kwargs):
    """Translate ``key`` for ``lang``, falling back to English.

    Any missing key or language degrades gracefully instead of raising,
    so a partially translated catalog can never crash the UI.
    """
    entry = TRANSLATIONS.get(lang) or TRANSLATIONS[DEFAULT_LANGUAGE]
    text = entry.get(key)
    if text is None:
        text = TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text
