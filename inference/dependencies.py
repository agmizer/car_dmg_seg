from typing import Optional

from inference.settings import EnvironmentConfig, InferenceConfig, VideoConfig, LiveKitConfig
from inference.services import InferenceService, LiveKitService


class Dependencies:
    """Singleton container for application dependencies"""

    _instance: Optional["Dependencies"] = None

    def __init__(self):
        """Initialize dependencies (should only be called once)"""
        if Dependencies._instance is not None:
            raise RuntimeError("Dependencies is a singleton. Use get_instance() instead.")

        # Load configuration
        self.config = EnvironmentConfig()

        # Initialize services
        self.inference_service = InferenceService(
            inference_config=self.config.inference,
            video_config=self.config.video
        )

        self.livekit_service = LiveKitService(
            livekit_config=self.config.livekit,
            inference_service=self.inference_service,
            video_config=self.config.video
        )

        Dependencies._instance = self

    @classmethod
    def get_instance(cls) -> "Dependencies":
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)"""
        cls._instance = None


def get_dependencies() -> Dependencies:
    """Helper function to get dependencies instance"""
    return Dependencies.get_instance()
