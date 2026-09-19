from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class PhoneVisualizer:
    """Draws phone boxes/status. Kept separate from visualization/main.py's
    ReportVisualizer because that class' coordinate-stacking logic is tightly
    coupled to the 6 existing drowsiness features; adding a 7th slot there
    would touch update_coordinates()'s indexing. This class instead owns a
    single fixed status line placed just below the lowest existing one
    (pitch/yawn sit at y=340/420 with 80px spacing, so phone uses y=500) and
    draws bounding boxes directly on the frame(s) it's given.
    """

    def __init__(self, status_position: Tuple[int, int] = (10, 500)):
        self.status_position = status_position

    @staticmethod
    def _color(possible_call: bool) -> Tuple[int, int, int]:
        # BGR: red for a likely call, yellow for a phone that's just visible
        return (0, 0, 255) if possible_call else (0, 255, 255)

    def draw_boxes(self, image: np.ndarray, boxes: List[Dict[str, Any]], possible_call: bool) -> np.ndarray:
        color = self._color(possible_call)
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"phone {box['confidence']:.2f}"
            label_y = max(15, y1 - 8)
            cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image

    def draw_status(self, image: np.ndarray, phone_present: bool, possible_call: bool) -> np.ndarray:
        if possible_call:
            text, color = "phone: possible call", (0, 0, 255)
        elif phone_present:
            text, color = "phone: visible", (0, 255, 255)
        else:
            text, color = "phone: not detected", (0, 255, 0)
        cv2.putText(image, text, self.status_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return image