from .logger import log, setup_logging
from .model_impl import My_LicensePlate_Model
from .video_mode import process_video

__version__ = "0.1.0"
__all__ = [
    "log",
    "setup_logging",
    "My_LicensePlate_Model",
    "process_video",
]