import os

# Project Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PARKING_COORDS_DIR = os.path.join(DATA_DIR, "parking_coords")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PARKING_COORDS_DIR, exist_ok=True)

# Database file
DB_PATH = os.path.join(DATA_DIR, "traffic_dashboard.db")
APP_SETTINGS_PATH = os.path.join(DATA_DIR, "dashboard_settings.json")
MAPPLS_API_KEY = os.getenv("MAPPLS_API_KEY", "").strip()

# General Settings
VIOLATION_ALERT_THRESHOLD = 3  # Alert if a plate is seen 3+ times
CONFIDENCE_THRESHOLD = 0.45    # YOLOv8 default confidence
OCR_CONFIDENCE_THRESHOLD = 0.3 # EasyOCR confidence
DEFAULT_FRAME_SKIP = 6
DEFAULT_PARKING_VIOLATION_SECONDS = 5.0
DEFAULT_WRONG_SIDE_MIN_MOVE = 15
DEFAULT_TRIPLE_OVERLAP_RATIO = 0.3
DEFAULT_HELMET_SKIN_RATIO = 0.15
DEFAULT_FRAME_WIDTH = 800
DEFAULT_FRAME_HEIGHT = 500

# Pre-configured parking coordinates (fallback defaults)
DEFAULT_PARKING_COORDS = [
    [36, 295, 169, 346],
    [100, 326, 251, 404],
    [486, 262, 559, 301],
    [565, 270, 625, 303],
    [437, 314, 600, 396],
    [248, 318, 427, 402]
]

# Detection Class Color Settings (BGR format for OpenCV)
COLORS = {
    "NORMAL": (0, 255, 0),        # Green
    "VIOLATION": (0, 0, 255),     # Red
    "PLATE": (0, 255, 255),       # Yellow (Cyan-like in BGR)
    "TEXT": (255, 255, 255)       # White
}

# Standard Violation Types
VIOLATION_TYPES = {
    "HELMET": "Helmet Violation",
    "TRIPLE_RIDING": "Triple Riding",
    "ILLEGAL_PARKING": "Illegal Parking",
    "WRONG_SIDE": "Wrong-side Driving",
    "SIGNAL_VIOLATION": "Red-light Violation"
}

# Default camera locations for Dehradun / NCR simulation
DEFAULT_CAMERAS = [
    {
        "id": "CAM-01",
        "location": "Mall Road Intersection",
        "latitude": 30.3218,
        "longitude": 78.0464
    },
    {
        "id": "CAM-02",
        "location": "Main Market Crossing",
        "latitude": 30.3165,
        "longitude": 78.0322
    },
    {
        "id": "CAM-03",
        "location": "Highway Bypass",
        "latitude": 30.3340,
        "longitude": 78.0580
    },
    {
        "id": "CAM-04",
        "location": "Residential Avenue Entrance",
        "latitude": 30.3080,
        "longitude": 78.0250
    }
]
