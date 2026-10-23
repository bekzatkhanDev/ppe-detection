"""Global configuration for PPE Safety Monitor."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Camera sources: 0 for webcam, or path/URL for video files/streams
DEFAULT_CAMERA = 0

# Detection classes expected from the trained YOLO model
CLASS_PERSON = "person"
CLASS_HELMET = "helmet"
CLASS_VEST = "vest"
CLASS_NO_HELMET = "no-helmet"
CLASS_NO_VEST = "no-vest"

# Colors (BGR for OpenCV drawing)
COLOR_SAFE = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_PERSON = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)

# Thresholds
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
OVERLAP_IOU = 0.3  # IoU between person and PPE to consider "worn"

# UI
WINDOW_TITLE = "PPE Safety Monitor"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_TARGET = 30

