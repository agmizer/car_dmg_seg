import asyncio
import signal
import sys
from typing import Optional

from inference.dependencies import get_dependencies


class InferenceServer:
    """Main inference server application"""

    def __init__(self):
        self.deps = get_dependencies()
        self.shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the inference server"""
        print("=" * 60)
        print("YOLO Segmentation Inference Server")
        print("=" * 60)
        print(f"Model: {self.deps.config.inference.model_path}")
        print(f"Device: {self.deps.config.inference.device}")
        print(f"Confidence threshold: {self.deps.config.inference.confidence_threshold}")
        print(f"Room: {self.deps.config.livekit.room_name}")
        print(f"LiveKit URL: {self.deps.config.livekit.url}")
        print("=" * 60)

        try:
            # Connect to LiveKit room
            await self.deps.livekit_service.connect()

            print("\nInference server is running. Press Ctrl+C to stop.")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            print(f"Error running inference server: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the inference server"""
        print("\nStopping inference server...")
        await self.deps.livekit_service.disconnect()
        print("Inference server stopped.")

    def handle_shutdown(self, sig, frame) -> None:
        """Handle shutdown signals"""
        print(f"\nReceived signal {sig}, shutting down...")
        self.shutdown_event.set()


async def async_main() -> None:
    """Async main entry point"""
    server = InferenceServer()

    # Register signal handlers
    signal.signal(signal.SIGINT, server.handle_shutdown)
    signal.signal(signal.SIGTERM, server.handle_shutdown)

    await server.start()


def run() -> None:
    """Main entry point for the application"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
