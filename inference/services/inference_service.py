import time
from pathlib import Path
from typing import List
import numpy as np
import cv2
from ultralytics import YOLO

from inference.models import DetectionResult, BoundingBox, SegmentationMask
from inference.settings import InferenceConfig, VideoConfig


class InferenceService:
    """Service for running YOLO segmentation inference"""

    def __init__(self, inference_config: InferenceConfig, video_config: VideoConfig):
        self.config = inference_config
        self.video_config = video_config
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model from the specified path"""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(str(model_path))
        print(f"Model loaded successfully on device: {self.config.device}")

    def infer(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single frame

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            DetectionResult containing bounding boxes, masks, and annotated frame
        """
        start_time = time.perf_counter()

        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            device=self.config.device,
            imgsz=self.config.imgsz,
            verbose=False
        )[0]

        # Extract bounding boxes
        bounding_boxes: List[BoundingBox] = []
        masks: List[SegmentationMask] = []

        if results.boxes is not None:
            for idx, box in enumerate(results.boxes):
                box_data = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]

                bbox = BoundingBox(
                    x1=float(box_data[0]),
                    y1=float(box_data[1]),
                    x2=float(box_data[2]),
                    y2=float(box_data[3]),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                bounding_boxes.append(bbox)

                # Extract corresponding mask if available
                if results.masks is not None and idx < len(results.masks):
                    mask_data = results.masks[idx].data[0].cpu().numpy()

                    seg_mask = SegmentationMask(
                        mask=mask_data,
                        bbox=bbox
                    )
                    masks.append(seg_mask)

        # Create annotated frame
        annotated_frame = self._annotate_frame(frame, bounding_boxes, masks)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return DetectionResult(
            bounding_boxes=bounding_boxes,
            masks=masks,
            annotated_frame=annotated_frame,
            inference_time_ms=inference_time_ms
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        bounding_boxes: List[BoundingBox],
        masks: List[SegmentationMask]
    ) -> np.ndarray:
        """
        Annotate frame with bounding boxes and segmentation masks

        Args:
            frame: Input frame
            bounding_boxes: List of detected bounding boxes
            masks: List of segmentation masks

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Define colors for different classes (you can customize these)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        # Draw masks first (so boxes are on top)
        for seg_mask in masks:
            mask = seg_mask.mask
            bbox = seg_mask.bbox
            color = colors[bbox.class_id % len(colors)]

            # Resize mask to frame size if needed
            if mask.shape != (annotated.shape[0], annotated.shape[1]):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (annotated.shape[1], annotated.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Create colored mask overlay
            mask_bool = mask > 0.5
            overlay = annotated.copy()
            overlay[mask_bool] = [
                int(overlay[mask_bool, i] * (1 - self.video_config.mask_alpha) +
                    color[i] * self.video_config.mask_alpha)
                for i in range(3)
            ]
            annotated = overlay

        # Draw bounding boxes and labels
        for bbox in bounding_boxes:
            color = colors[bbox.class_id % len(colors)]

            # Draw rectangle
            cv2.rectangle(
                annotated,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                2
            )

            # Prepare label
            label = f"{bbox.class_name} {bbox.confidence:.2f}"

            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background
            cv2.rectangle(
                annotated,
                (int(bbox.x1), int(bbox.y1) - label_height - baseline - 5),
                (int(bbox.x1) + label_width, int(bbox.y1)),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (int(bbox.x1), int(bbox.y1) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return annotated
