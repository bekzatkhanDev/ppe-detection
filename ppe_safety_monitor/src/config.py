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
CLASS_GLOVES = "gloves"
CLASS_GOGGLES = "goggles"
CLASS_BOOTS = "boots"
CLASS_NO_HELMET = "no_helmet"
CLASS_NO_VEST = "no_vest"
CLASS_NO_GLOVES = "no_gloves"
CLASS_NO_GOGGLES = "no_goggles"
# The bundled dataset uses the singular spelling "no_goggle".
CLASS_NO_GOGGLE_SINGULAR = "no_goggle"
CLASS_NO_BOOTS = "no_boots"
# Class used by some models for a bare worker without any protective gear.
CLASS_NONE = "none"

# All PPE gear the app can monitor (order used in the Settings dialog).
PPE_CLASSES = [CLASS_HELMET, CLASS_VEST, CLASS_GLOVES, CLASS_GOGGLES, CLASS_BOOTS]

# Default PPE items that must be worn for a person to be considered safe.
# Overridable at runtime via settings.json ("ppe_to_monitor").
REQUIRED_PPE = [CLASS_HELMET, CLASS_VEST]

# Map of native "missing gear" YOLO classes -> the PPE item they confirm.
# Used as secondary evidence when the anatomical-zone heuristic finds no gear.
NO_PPE_MAP = {
    CLASS_NO_HELMET: CLASS_HELMET,
    CLASS_NO_VEST: CLASS_VEST,
    CLASS_NO_GLOVES: CLASS_GLOVES,
    CLASS_NO_GOGGLES: CLASS_GOGGLES,
    CLASS_NO_GOGGLE_SINGULAR: CLASS_GOGGLES,
    CLASS_NO_BOOTS: CLASS_BOOTS,
}

# Keys of the per-frame statistics dict produced by the inference worker and
# consumed by the UI. Shared constants avoid implicit string coupling.
STAT_KEY_PERSONS = "persons"
STAT_KEY_SAFE = "safe"
STAT_KEY_UNSAFE = "unsafe"
STAT_KEY_MISSING_HELMET = "missing_helmet"
STAT_KEY_MISSING_VEST = "missing_vest"
STAT_KEY_MISSING_GLOVES = "missing_gloves"
STAT_KEY_MISSING_GOGGLES = "missing_goggles"
STAT_KEY_MISSING_BOOTS = "missing_boots"
STAT_KEYS = (STAT_KEY_PERSONS, STAT_KEY_SAFE, STAT_KEY_UNSAFE,
             STAT_KEY_MISSING_HELMET, STAT_KEY_MISSING_VEST,
             STAT_KEY_MISSING_GLOVES, STAT_KEY_MISSING_GOGGLES,
             STAT_KEY_MISSING_BOOTS)

# Colors (BGR for OpenCV drawing)
COLOR_SAFE = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)

# Thresholds
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
OVERLAP_RATIO = 0.5  # fraction of the PPE box inside the person box to count as "worn"

# Size/noise filtering: YOLO often hallucinates tiny boxes for small gear
# (gloves/boots) from background texture. Any gear box smaller than this
# (width OR height, in pixels) is ignored during detection parsing.
MIN_GEAR_BOX_SIZE = 15

# ---- Settings persistence (settings.json in the app root) -----------------
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "camera_sources": [DEFAULT_CAMERA],
    "ppe_to_monitor": list(REQUIRED_PPE),
}


def get_asset_path(*relative_parts):
    """Return a path inside the app directory.

    When frozen with PyInstaller, resources bundled by the spec live under
    ``sys._MEIPASS``, but *writable* user data (settings.json, ppe_app.db)
    must live next to the executable so it persists between runs. This
    helper resolves the correct base for each case:
      * frozen exe   -> directory containing the executable
      * plain script -> project BASE_DIR
    """
    import sys
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = BASE_DIR
    return os.path.join(base, *relative_parts)


def load_settings():
    """Load settings.json; fall back to sensible defaults when missing/corrupt."""
    import json
    path = get_asset_path("settings.json")
    settings = {
        "camera_sources": list(DEFAULT_SETTINGS["camera_sources"]),
        "ppe_to_monitor": list(DEFAULT_SETTINGS["ppe_to_monitor"]),
    }
    if not os.path.exists(path):
        return settings
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cams = data.get("camera_sources")
        if isinstance(cams, list) and cams:
            settings["camera_sources"] = [int(c) for c in cams]
        ppe = data.get("ppe_to_monitor")
        if isinstance(ppe, list):
            allowed = {c.lower() for c in PPE_CLASSES}
            cleaned = [str(p).strip().lower() for p in ppe
                       if str(p).strip().lower() in allowed]
            settings["ppe_to_monitor"] = cleaned
    except (OSError, ValueError, TypeError):
        pass  # corrupt/unreadable file -> keep defaults
    return settings


def save_settings(data):
    """Atomically write settings dict to settings.json in the app root."""
    import json
    path = get_asset_path("settings.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)
    return path

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
        # Settings dialog / dynamic PPE monitoring.
        "settings": "⚙️ Settings",
        "missing_gloves": "Missing Gloves",
        "missing_goggles": "Missing Goggles",
        "missing_boots": "Missing Boots",
        "ov_gloves": "Gloves",
        "ov_boots": "Boots",
        "ov_goggles": "Goggles",
        "ov_no_gloves": "no gloves",
        "ov_no_boots": "no boots",
        "ov_no_goggles": "no goggles",
        "ov_no_gloves_box": "No Gloves",
        "ov_no_boots_box": "No Boots",
        "ov_no_goggles_box": "No Goggles",
        "settings_applied": "Settings applied",
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
        "settings": "⚙️ Настройки",
        "missing_gloves": "Нет перчаток",
        "missing_goggles": "Нет очков",
        "missing_boots": "Нет ботинк",
        "ov_gloves": "Перчатки",
        "ov_boots": "Ботинки",
        "ov_goggles": "Очки",
        "ov_no_gloves": "нет перчаток",
        "ov_no_boots": "нет ботинок",
        "ov_no_goggles": "нет очков",
        "ov_no_gloves_box": "Нет перчаток",
        "ov_no_boots_box": "Нет ботинок",
        "ov_no_goggles_box": "Нет очков",
        "settings_applied": "Настройки применены",
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
        "settings": "⚙️ Параметрлер",
        "missing_gloves": "Қолғап жоқ",
        "missing_goggles": "Көзілдірік жоқ",
        "missing_boots": "Бәкімсе жоқ",
        "ov_gloves": "Қолғап",
        "ov_boots": "Бәкімсе",
        "ov_goggles": "Көзілдірік",
        "ov_no_gloves": "қолғап жоқ",
        "ov_no_boots": "бәкімсе жоқ",
        "ov_no_goggles": "көзілдірік жоқ",
        "ov_no_gloves_box": "Қолғап жоқ",
        "ov_no_boots_box": "Бәкімсе жоқ",
        "ov_no_goggles_box": "Көзілдірік жоқ",
        "settings_applied": "Параметрлер қолданылды",
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
