"""
YOLO segmentation inference demo with ByteTrack tracking.

Usage:
    python demo.py --camera 0
    python demo.py --camera 1 --no-tracking
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from settings import config

COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]


@dataclass
class TrackResult:
    """Single tracked object result."""

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    is_new: bool
    mask: Optional[np.ndarray] = None


class ByteTrackWrapper:
    """Wrapper around Supervision's ByteTrack for multi-object tracking."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        """
        Initialize ByteTrack wrapper.

        Args:
            track_activation_threshold: Confidence threshold for new tracks.
            lost_track_buffer: Frames to keep track alive without detection.
            minimum_matching_threshold: IoU threshold for matching.
            frame_rate: Expected frame rate for track buffer calculation.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self._seen_track_ids: set[int] = set()

    def update(
        self,
        detections: list[tuple[int, int, int, int, float, int]],
        masks: Optional[list[np.ndarray]] = None,
    ) -> list[TrackResult]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, confidence, class_id) tuples.
            masks: Optional segmentation masks corresponding to detections.

        Returns:
            List of TrackResult with assigned track IDs.
        """
        if not detections:
            return []

        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confidence = np.array([d[4] for d in detections])
        class_id = np.array([d[5] for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked_detections = self.tracker.update_with_detections(sv_detections)

        results = []
        if tracked_detections.tracker_id is None:
            return results

        for i, track_id in enumerate(tracked_detections.tracker_id):
            track_id = int(track_id)
            bbox = tuple(map(int, tracked_detections.xyxy[i]))
            conf = (
                float(tracked_detections.confidence[i])
                if tracked_detections.confidence is not None
                else 0.0
            )
            cls_id = (
                int(tracked_detections.class_id[i])
                if tracked_detections.class_id is not None
                else 0
            )

            is_new = track_id not in self._seen_track_ids
            if is_new:
                self._seen_track_ids.add(track_id)

            mask = None
            if masks is not None:
                for orig_idx, det in enumerate(detections):
                    orig_bbox = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                    if self._bbox_iou(bbox, orig_bbox) > 0.9:
                        if orig_idx < len(masks):
                            mask = masks[orig_idx]
                        break

            results.append(
                TrackResult(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    is_new=is_new,
                    mask=mask,
                )
            )

        return results

    def _bbox_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def reset(self):
        """Reset the tracker state."""
        self.tracker.reset()
        self._seen_track_ids.clear()

    @property
    def active_track_count(self) -> int:
        """Number of track IDs seen so far."""
        return len(self._seen_track_ids)


def draw_results_with_tracking(
    frame: np.ndarray,
    track_results: list[TrackResult],
    model: YOLO,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """Draw bounding boxes and segmentation masks with track IDs on frame."""
    annotated = frame.copy()

    for track in track_results:
        if track.mask is not None:
            color = COLORS[track.track_id % len(COLORS)]
            mask_resized = cv2.resize(
                track.mask.astype(np.float32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bool = mask_resized > 0.5
            color_array = np.array(color, dtype=np.float32)
            annotated[mask_bool] = (
                annotated[mask_bool].astype(np.float32) * (1 - mask_alpha)
                + color_array * mask_alpha
            ).astype(np.uint8)

    for track in track_results:
        x1, y1, x2, y2 = track.bbox
        color = COLORS[track.track_id % len(COLORS)]
        class_name = model.names.get(track.class_id, f"class_{track.class_id}")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {class_name} {track.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if track.is_new:
            cv2.putText(
                annotated,
                "NEW",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return annotated


def draw_results_no_tracking(
    frame: np.ndarray,
    results,
    model: YOLO,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """Draw bounding boxes and segmentation masks on frame without tracking."""
    annotated = frame.copy()

    if results.boxes is None:
        return annotated

    if results.masks is not None:
        for idx, mask in enumerate(results.masks):
            if idx >= len(results.boxes):
                break

            class_id = int(results.boxes[idx].cls[0].cpu().numpy())
            color = COLORS[class_id % len(COLORS)]

            mask_data = mask.data[0].cpu().numpy()
            mask_resized = cv2.resize(
                mask_data.astype(np.float32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bool = mask_resized > 0.5
            color_array = np.array(color, dtype=np.float32)
            annotated[mask_bool] = (
                annotated[mask_bool].astype(np.float32) * (1 - mask_alpha)
                + color_array * mask_alpha
            ).astype(np.uint8)

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[class_id]
        color = COLORS[class_id % len(COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated


def main():
    """Run the YOLO segmentation demo with optional ByteTrack tracking."""
    parser = argparse.ArgumentParser(
        description="YOLO Segmentation Webcam Demo with ByteTrack"
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=config.inference.model_path,
        help=f"Path to YOLO model (default: {config.inference.model_path})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.inference.confidence_threshold,
        help=f"Confidence threshold (default: {config.inference.confidence_threshold})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.inference.device,
        help=f"Device to run on (default: {config.inference.device})",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable object tracking",
    )
    parser.add_argument(
        "--lost-buffer",
        type=int,
        default=30,
        help="Frames to keep track alive without detection (default: 30)",
    )

    args = parser.parse_args()
    use_tracking = not args.no_tracking

    print("=" * 60)
    print("YOLO Segmentation Demo" + (" with ByteTrack" if use_tracking else ""))
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    print(f"Camera: {args.camera}")
    print(f"Tracking: {'Enabled' if use_tracking else 'Disabled'}")
    if use_tracking:
        print(f"Lost buffer: {args.lost_buffer} frames")
    print("=" * 60)

    print("Loading model...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1

    model = YOLO(str(model_path))
    print(f"Model loaded: {len(model.names)} classes")
    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

    tracker = None
    if use_tracking:
        print("\nInitializing ByteTrack tracker...")
        tracker = ByteTrackWrapper(
            track_activation_threshold=args.conf,
            lost_track_buffer=args.lost_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        for idx in range(5):
            if idx == args.camera:
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Using camera index {idx}")
                break
        else:
            print("ERROR: Could not open any camera")
            return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {width}x{height}")
    print("\nPress 'q' to quit, 'r' to reset tracker")
    print("=" * 60)

    frame_count = 0
    fps_start = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            inference_start = time.perf_counter()
            results = model(
                frame,
                conf=args.conf,
                device=args.device,
                imgsz=config.inference.imgsz,
                verbose=False,
            )[0]
            inference_time = (time.perf_counter() - inference_start) * 1000

            if use_tracking and tracker is not None:
                detections = []
                masks = []

                if results.boxes is not None:
                    for idx, box in enumerate(results.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        detections.append(
                            (int(x1), int(y1), int(x2), int(y2), confidence, class_id)
                        )

                        if results.masks is not None and idx < len(results.masks):
                            mask_data = results.masks[idx].data[0].cpu().numpy()
                            masks.append(mask_data)
                        else:
                            masks.append(None)

                track_results = tracker.update(detections, masks if masks else None)
                annotated = draw_results_with_tracking(
                    frame, track_results, model, config.video.mask_alpha
                )
                num_detections = len(track_results)
                total_tracks = tracker.active_track_count
            else:
                annotated = draw_results_no_tracking(
                    frame, results, model, config.video.mask_alpha
                )
                num_detections = len(results.boxes) if results.boxes is not None else 0
                total_tracks = 0

            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()

            if use_tracking:
                stats = f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms | Tracks: {num_detections} | Total: {total_tracks}"
            else:
                stats = f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms | Detections: {num_detections}"

            cv2.putText(
                annotated,
                stats,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            window_title = (
                "YOLO Segmentation"
                + (" + ByteTrack" if use_tracking else "")
                + " (press 'q' to quit)"
            )
            cv2.imshow(window_title, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuit requested")
                break
            elif key == ord("r") and tracker is not None:
                print("\nResetting tracker...")
                tracker.reset()

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
