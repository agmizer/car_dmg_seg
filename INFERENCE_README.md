# YOLO Segmentation Inference Server

Real-time car damage segmentation inference server using LiveKit and YOLO11.

## Architecture

```
inference/
├── services/
│   ├── __init__.py
│   ├── inference_service.py    # YOLO model inference
│   └── livekit_service.py      # LiveKit room & participant handling
├── models/
│   ├── __init__.py
│   └── detection.py            # Pydantic models for detections
├── main.py                     # Entry point
├── dependencies.py             # Singleton instances
└── settings.py                 # Pydantic BaseSettings for env vars
```

## Features

- **Real-time Processing**: Processes video frames from all participants in a LiveKit room
- **YOLO11 Segmentation**: Uses your trained car damage segmentation model
- **Configurable**: All settings managed via environment variables
- **Multi-participant**: Automatically handles multiple participants joining/leaving
- **Annotated Output**: Publishes video streams with segmentation masks and bounding boxes

## Installation

Dependencies are already in `pyproject.toml`. Install with:

```bash
uv sync
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your LiveKit credentials and preferences:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_ROOM_NAME=car-damage-detection

# Inference Configuration
INFERENCE_MODEL_PATH=yolo/runs/segment/car_damage_seg4/weights/best.pt
INFERENCE_CONFIDENCE_THRESHOLD=0.25
INFERENCE_DEVICE=cuda
INFERENCE_IMGSZ=640

# Video Configuration
VIDEO_FPS=30
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
VIDEO_MASK_ALPHA=0.4
```

### Configuration Options

#### LiveKit Settings
- `LIVEKIT_URL`: Your LiveKit server URL
- `LIVEKIT_API_KEY`: LiveKit API key
- `LIVEKIT_API_SECRET`: LiveKit API secret
- `LIVEKIT_ROOM_NAME`: Room name to join

#### Inference Settings
- `INFERENCE_MODEL_PATH`: Path to YOLO model weights (default: best.pt from car_damage_seg4)
- `INFERENCE_CONFIDENCE_THRESHOLD`: Minimum confidence for detections (0.0-1.0)
- `INFERENCE_DEVICE`: Device for inference (`cuda` or `cpu`)
- `INFERENCE_IMGSZ`: Input image size for YOLO

#### Video Settings
- `VIDEO_FPS`: Target FPS for output stream
- `VIDEO_WIDTH`: Output video width
- `VIDEO_HEIGHT`: Output video height
- `VIDEO_MASK_ALPHA`: Transparency of segmentation masks (0.0=transparent, 1.0=opaque)

## Usage

### Using the script entry point

```bash
inference-server
```

### Using Python module

```bash
python -m inference.main
```

### What happens

1. Server connects to the specified LiveKit room
2. When participants join and publish video tracks:
   - Server subscribes to their video
   - Runs YOLO segmentation on every frame
   - Publishes annotated video back to the room (track name: `{participant_identity}_annotated`)
3. Multiple participants are processed concurrently
4. Press Ctrl+C to gracefully shutdown

## Output

For each participant, the server publishes a new video track with:
- **Segmentation masks**: Colored semi-transparent overlays on detected damage
- **Bounding boxes**: Rectangles around detected regions
- **Labels**: Class name and confidence score

## Development

### Project Structure

- `settings.py`: Uses Pydantic BaseSettings with subtypes for each configuration section
- `models/`: Pydantic BaseModel classes for strong typing
- `services/inference_service.py`: Handles YOLO model loading and inference
- `services/livekit_service.py`: Manages LiveKit connection and participant processing
- `dependencies.py`: Singleton pattern for service instances
- `main.py`: Application entry point with signal handling

### Key Classes

#### `InferenceService`
- Loads YOLO model
- Runs inference on frames
- Annotates frames with masks and bounding boxes

#### `LiveKitService`
- Connects to LiveKit room
- Manages participant lifecycle
- Creates `ParticipantProcessor` for each participant

#### `ParticipantProcessor`
- Processes video from one participant
- Converts frames to numpy arrays
- Publishes annotated frames back to room

## Troubleshooting

### CUDA/GPU Issues
If you encounter CUDA errors, set device to CPU:
```env
INFERENCE_DEVICE=cpu
```

### Model Not Found
Ensure the model path is correct relative to the project root:
```env
INFERENCE_MODEL_PATH=yolo/runs/segment/car_damage_seg4/weights/best.pt
```

### LiveKit Connection Issues
- Verify your LiveKit URL, API key, and secret
- Check that the room name doesn't have special characters
- Ensure firewall allows WebSocket connections

### Performance
- Reduce `INFERENCE_IMGSZ` for faster inference (e.g., 320)
- Increase `INFERENCE_CONFIDENCE_THRESHOLD` to filter out low-confidence detections
- Use a smaller YOLO model (e.g., yolo11n-seg.pt instead of yolo11m-seg.pt)

## Example Client Setup

To test the server, you can use a simple LiveKit client that publishes video. Here's a basic example using the LiveKit Python SDK:

```python
import asyncio
from livekit import rtc

async def main():
    room = rtc.Room()

    await room.connect(
        "wss://your-server.livekit.cloud",
        "your_token_here"
    )

    # Publish your camera or video file
    # The inference server will process it and publish annotated video

    await room.disconnect()

asyncio.run(main())
```

## License

This project uses YOLO11 from Ultralytics. Please comply with their license terms.
