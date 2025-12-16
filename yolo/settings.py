from pydantic import Field
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent.absolute()
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

class InferenceConfig(BaseSettings):
    """YOLO inference settings"""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_")

    model_path: str = Field(
        default="yolo/runs/segment/car_damage_seg4/weights/best.pt",
        description="Path to YOLO model weights"
    )
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on (cuda/cpu)"
    )
    imgsz: int = Field(
        default=640,
        description="Input image size for inference"
    )


class VideoConfig(BaseSettings):
    """Video processing settings"""

    model_config = SettingsConfigDict(env_prefix="VIDEO_")

    fps: int = Field(
        default=30,
        description="Target FPS for output video stream"
    )
    width: int = Field(
        default=1280,
        description="Output video width"
    )
    height: int = Field(
        default=720,
        description="Output video height"
    )
    mask_alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Transparency of segmentation masks (0=transparent, 1=opaque)"
    )

# Singleton config instance
config = EnvironmentConfig()