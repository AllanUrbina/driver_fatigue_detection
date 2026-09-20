from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO

# COCO class id for "cell phone" in the stock 80-class dataset that the
# pretrained yolov8s.pt weights are already trained on. Because this class
# is already covered, no custom dataset or training run is required to get
# phone detection working: the pretrained checkpoint is downloaded
# automatically by ultralytics the first time PhoneDetector() runs.
CELL_PHONE_CLASS_ID = 67


class PhoneDetector:
    """Finds cell phones in a single video frame using YOLOv8n.

    This mirrors the role of FaceMeshProcessor/HandsProcessor in
    extract_points/: it only extracts raw detections from the frame, it does
    not decide whether that counts as "phone use" (that debouncing/logic
    lives in phone_detection/processing.py, same split as the rest of the
    pipeline).
    """

    def __init__(self, model_path: str = "yolov8s.pt", confidence_threshold: float = 0.55,
                 image_size: int = 480):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            frame,
            classes=[CELL_PHONE_CLASS_ID],
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": confidence,
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                })
        return detections
