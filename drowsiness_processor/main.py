import numpy as np
import base64
import cv2

from drowsiness_processor.extract_points.point_extractor import PointsExtractor
from drowsiness_processor.data_processing.main import PointsProcessing
from drowsiness_processor.drowsiness_features.processing import FeaturesDrowsinessProcessing
from drowsiness_processor.visualization.main import ReportVisualizer
from drowsiness_processor.reports.main import DrowsinessReports
from drowsiness_processor.phone_detection.detector import PhoneDetector
from drowsiness_processor.phone_detection.processing import PhoneUseEstimator
from drowsiness_processor.phone_detection.visualization import PhoneVisualizer


class DrowsinessDetectionSystem:
    def __init__(self):
        self.points_extractor = PointsExtractor()
        self.points_processing = PointsProcessing()
        self.features_processing = FeaturesDrowsinessProcessing()
        self.visualizer = ReportVisualizer()
        self.reports = DrowsinessReports('drowsiness_processor/reports/august/drowsiness_report.csv')
        self.phone_detector = PhoneDetector()
        self.phone_estimator = PhoneUseEstimator()
        self.phone_visualizer = PhoneVisualizer()
        self.json_report: dict = {}

    def run(self, picture_base64: str):
        # decode base64
        picture_bytes = base64.b64decode(picture_base64)
        # convert bytes to OpenCV image
        picture = cv2.imdecode(np.frombuffer(picture_bytes, np.uint8), cv2.IMREAD_COLOR)
        return self.frame_processing(picture)

    def frame_processing(self, face_image: np.ndarray):
        key_points, control_process, sketch = self.points_extractor.process(face_image)

        # phone detection is intentionally NOT gated behind control_process:
        # a distracted driver looking away is exactly the case where the
        # face mesh can fail while a phone is still visible in frame.
        head_points: dict = {}
        drowsiness_features_processed: dict = {}
        if control_process:
            points_processed = self.points_processing.main(key_points)
            head_points = points_processed.get('head', {})
            drowsiness_features_processed = self.features_processing.main(points_processed)

        phone_boxes = self.phone_detector.detect(face_image)
        phone_report = self.phone_estimator.process({'boxes': phone_boxes, 'head_points': head_points})
        drowsiness_features_processed['phone'] = phone_report

        phone_present = len(phone_boxes) > 0
        possible_call = self.phone_estimator.last_possible_call
        sketch = self.phone_visualizer.draw_boxes(sketch, phone_boxes, possible_call)
        sketch = self.phone_visualizer.draw_status(sketch, phone_present, possible_call)
        face_image = self.phone_visualizer.draw_boxes(face_image, phone_boxes, possible_call)

        if control_process:
            sketch = self.visualizer.visualize_all_reports(sketch, drowsiness_features_processed)

        self.reports.main(drowsiness_features_processed)
        self.json_report = self.reports.generate_json_report(drowsiness_features_processed)
        return face_image, sketch, self.json_report