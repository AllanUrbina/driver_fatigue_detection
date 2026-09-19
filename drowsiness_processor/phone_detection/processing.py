import time
from typing import Any, Dict, List, Tuple

import numpy as np

from drowsiness_processor.drowsiness_features.processor import DrowsinessProcessor


class PhoneProximityClassifier:
    """Approximates the docx proposal's 'posible llamada' rule: a phone is
    flagged as a possible call when its center is close to either cheek
    landmark relative to the driver's own face width, instead of a fixed
    pixel distance (which would break as the driver moves closer/farther
    from the camera). It reuses the right_cheek_point/left_cheek_point that
    head_processing.py already computes for the pitch/nod feature, so no new
    face landmarks need to be extracted.
    """

    def __init__(self, ear_proximity_ratio: float = 0.65):
        self.ear_proximity_ratio = ear_proximity_ratio

    def classify(self, phone_center: Tuple[int, int], head_points: dict) -> bool:
        if not head_points:
            return False

        right_cheek = head_points.get('right_cheek_point')
        left_cheek = head_points.get('left_cheek_point')
        if right_cheek is None or left_cheek is None:
            return False

        right_cheek = np.array(right_cheek, dtype=float)
        left_cheek = np.array(left_cheek, dtype=float)
        center = np.array(phone_center, dtype=float)

        face_width = np.linalg.norm(right_cheek - left_cheek)
        if face_width <= 0:
            return False

        nearest_distance = min(
            np.linalg.norm(center - right_cheek),
            np.linalg.norm(center - left_cheek),
        )
        return bool(nearest_distance <= face_width * self.ear_proximity_ratio)


class PhoneUseDetection:
    """Debounces raw per-frame phone detections into a sustained "phone use"
    event, the same start/flag/duration pattern used by PitchDetection,
    YawnDetection, etc. This absorbs single-frame false positives/negatives
    from YOLO instead of alerting on every flicker of the detector.
    """

    def __init__(self, sustained_seconds: float = 1.0):
        self.sustained_seconds = sustained_seconds
        self.start_time: float = 0
        self.flag: bool = False

    def detect(self, phone_present: bool) -> Tuple[bool, float]:
        if phone_present and not self.flag:
            self.start_time = time.time()
            self.flag = True
        elif not phone_present and self.flag:
            end_time = time.time()
            duration = round(end_time - self.start_time, 0)
            self.flag = False
            if duration >= self.sustained_seconds:
                self.start_time = 0
                return True, duration
        return False, 0.0


class PhoneUseCounter:
    def __init__(self):
        self.phone_count: int = 0
        self.phone_durations: List[str] = []

    def increment(self, duration: float, possible_call: bool):
        self.phone_count += 1
        label = 'possible call' if possible_call else 'phone visible'
        self.phone_durations.append(f"{self.phone_count} {label}: {duration} seconds")

    def reset(self):
        self.phone_count = 0
        self.phone_durations = []

    def get_durations(self) -> List[str]:
        return self.phone_durations


class PhoneUseReportGenerator:
    def generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'phone_report': data.get('phone_report', False),
            'phone_count': data.get('phone_count', 0),
            'phone_durations': data.get('phone_durations', []),
            'possible_call': data.get('possible_call', False),
        }


class PhoneUseEstimator(DrowsinessProcessor):
    """Turns raw YOLOv8n boxes + head landmarks into a debounced phone-use
    event. Same role in this pipeline as PitchEstimator/YawnEstimator/etc:
    a DrowsinessProcessor that main.py's FeaturesDrowsinessProcessing-style
    orchestration can call every frame.
    """

    def __init__(self, sustained_seconds: float = 1.0, ear_proximity_ratio: float = 0.65):
        self.detection = PhoneUseDetection(sustained_seconds)
        self.proximity_classifier = PhoneProximityClassifier(ear_proximity_ratio)
        self.counter = PhoneUseCounter()
        self.report_generator = PhoneUseReportGenerator()
        # exposed so the visualizer can draw the right color for the
        # current frame without recomputing the classification itself
        self.last_possible_call: bool = False
        # latched across a whole sustained event: by the time the phone
        # disappears and the event fires, last_possible_call has already
        # gone back to False for this (empty) frame, so the event-level
        # report needs its own memory of "was it ever a call while held"
        self._event_possible_call: bool = False

    def process(self, phone_input: dict):
        boxes: List[Dict[str, Any]] = phone_input.get('boxes', [])
        head_points: dict = phone_input.get('head_points', {})

        phone_present = len(boxes) > 0
        if phone_present:
            best_box = max(boxes, key=lambda box: box['confidence'])
            self.last_possible_call = self.proximity_classifier.classify(best_box['center'], head_points)
            if self.last_possible_call:
                self._event_possible_call = True
        else:
            self.last_possible_call = False

        is_event, duration = self.detection.detect(phone_present)
        if is_event:
            event_possible_call = self._event_possible_call
            self._event_possible_call = False
            self.counter.increment(duration, event_possible_call)
            return self.report_generator.generate_report({
                'phone_report': True,
                'phone_count': self.counter.phone_count,
                'phone_durations': self.counter.get_durations(),
                'possible_call': event_possible_call,
            })

        return {
            'phone_report': False,
            'phone_present': phone_present,
            'possible_call': self.last_possible_call,
        }