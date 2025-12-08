from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box coordinates"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class ID of detected object")
    class_name: str = Field(..., description="Class name of detected object")


class SegmentationMask(BaseModel):
    """Segmentation mask for a detected object"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mask: np.ndarray = Field(..., description="Binary mask array")
    bbox: BoundingBox = Field(..., description="Bounding box of the masked region")


class DetectionResult(BaseModel):
    """Complete detection result for a frame"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="Detected bounding boxes")
    masks: List[SegmentationMask] = Field(default_factory=list, description="Segmentation masks")
    annotated_frame: Optional[np.ndarray] = Field(None, description="Frame with annotations drawn")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
