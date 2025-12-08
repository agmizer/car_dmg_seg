from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveKitConfig(BaseSettings):
    """LiveKit connection settings"""

    model_config = SettingsConfigDict(env_prefix="LIVEKIT_")

    url: str = Field(..., description="LiveKit server URL (e.g., wss://your-server.livekit.cloud)")
    api_key: str = Field(..., description="LiveKit API key")
    api_secret: str = Field(..., description="LiveKit API secret")
    room_name: str = Field(..., description="LiveKit room name to join")


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


class EnvironmentConfig(BaseSettings):
    """Main configuration combining all settings"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    livekit: LiveKitConfig = Field(default_factory=LiveKitConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
