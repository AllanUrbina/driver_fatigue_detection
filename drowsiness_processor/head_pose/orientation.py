"""
Clasificador de orientación de la cabeza: "frente" | "izquierda" | "derecha" | "abajo".

Entrada : los 478 landmarks de MediaPipe FaceMesh (los mismos que ya extrae tu proyecto).
Salida  : HeadPoseResult (etiqueta + ángulos yaw/pitch en grados + caja del rostro).

Cómo funciona (resumen para el informe)
---------------------------------------
1. Se toman 6 landmarks que NO dependen de la iluminación ni de los lentes:
   punta de la nariz (1), mentón (152), comisuras externas de los ojos (33, 263) y comisuras
   de la boca (61, 291). Nunca se usan párpados ni iris, que son los puntos que los lentes
   y los reflejos alteran.
2. Con esos 6 puntos 2D y un modelo 3D genérico de rostro se resuelve la pose con
   cv2.solvePnP. De la pose se toma hacia dónde "apunta" la cara y se convierte en
   yaw (izquierda/derecha) y pitch (arriba/abajo) en grados.
3. CALIBRACIÓN AUTOMÁTICA: durante los primeros ~15 fotogramas el sistema asume que el
   conductor mira al frente y guarda ese ángulo como "cero". Así se corrige la posición
   de la cámara y las diferencias entre un rostro real y el modelo genérico.
4. Suavizado exponencial + histéresis: evita que la etiqueta parpadee cuando el ángulo
   está cerca del umbral (un parpadeo reiniciaría el cronómetro del semáforo).

Convención de signos (imagen SIN espejo, como sale de la cámara):
    yaw   > 0  -> la nariz se mueve hacia la DERECHA de la imagen = el conductor gira a SU IZQUIERDA
    pitch > 0  -> la cabeza baja
Con label_perspective="driver" (por defecto) "izquierda" significa la izquierda del conductor.
Si prefieres que signifique la izquierda de la pantalla usa label_perspective="screen".

Nota: los textos que OpenCV dibuja en el frame no admiten tildes ni ñ, por eso las etiquetas
("frente", "izquierda", "derecha", "abajo") van sin acentos.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FRENTE = "frente"
IZQUIERDA = "izquierda"
DERECHA = "derecha"
ABAJO = "abajo"
HEAD_LABELS = (FRENTE, IZQUIERDA, DERECHA, ABAJO)

# Índices FaceMesh usados (orden = orden de _MODEL_POINTS)
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_EYE_LEFT_IMG = 33      # comisura externa del ojo que aparece a la izquierda de la imagen
IDX_EYE_RIGHT_IMG = 263    # comisura externa del ojo que aparece a la derecha de la imagen
IDX_MOUTH_LEFT_IMG = 61
IDX_MOUTH_RIGHT_IMG = 291
_PNP_INDICES = (IDX_NOSE_TIP, IDX_CHIN, IDX_EYE_LEFT_IMG, IDX_EYE_RIGHT_IMG,
                IDX_MOUTH_LEFT_IMG, IDX_MOUTH_RIGHT_IMG)
_MIN_LANDMARKS = max(_PNP_INDICES) + 1

# Modelo 3D genérico (convención OpenCV: x a la derecha, y hacia ABAJO, z alejándose de la cámara).
# Origen en la punta de la nariz. Unidades arbitrarias (proporciones de un rostro promedio).
_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],         # 1   nariz
    [0.0, 330.0, 65.0],      # 152 mentón
    [-225.0, -170.0, 135.0],  # 33  ojo (lado izquierdo de la imagen)
    [225.0, -170.0, 135.0],   # 263 ojo (lado derecho de la imagen)
    [-150.0, 150.0, 125.0],   # 61  boca (lado izquierdo de la imagen)
    [150.0, 150.0, 125.0],    # 291 boca (lado derecho de la imagen)
], dtype=np.float64)
_MODEL_EYE_DISTANCE = 450.0  # distancia entre las comisuras de los ojos en el modelo


@dataclass
class HeadOrientationConfig:
    """Todos los parámetros ajustables en un solo lugar (ángulos en grados)."""
    yaw_threshold: float = 30.0          # |yaw| mayor a esto -> izquierda/derecha
    pitch_down_threshold: float = 22.0   # pitch mayor a esto -> abajo
    hysteresis: float = 0.75             # para SALIR de una etiqueta se exige bajar a umbral*0.75
    smoothing: float = 0.5               # 0<alpha<=1 ; 1 = sin suavizado
    label_perspective: str = "driver"    # "driver" | "screen"
    focal_length_factor: float = 1.0     # focal aproximada = factor * ancho de imagen
    min_eye_distance_px: float = 20.0    # rostros más pequeños se consideran no confiables
    # --- calibración automática ---
    auto_calibrate: bool = True
    calibration_frames: int = 15         # fotogramas "de frente" necesarios
    calibration_max_abs_yaw: float = 25.0    # solo cuentan fotogramas con |yaw crudo| <= esto
    calibration_max_abs_pitch: float = 25.0
    calibration_timeout_s: float = 10.0  # si no logra calibrar, usa el modelo genérico (0, 0)

    def __post_init__(self):
        if self.label_perspective not in ("driver", "screen"):
            raise ValueError("label_perspective debe ser 'driver' o 'screen'")
        if not (0.0 < self.smoothing <= 1.0):
            raise ValueError("smoothing debe estar en (0, 1]")
        if not (0.0 < self.hysteresis <= 1.0):
            raise ValueError("hysteresis debe estar en (0, 1]")
        if self.yaw_threshold <= 0 or self.pitch_down_threshold <= 0:
            raise ValueError("los umbrales deben ser positivos")


@dataclass
class HeadPoseResult:
    label: Optional[str] = None          # frente/izquierda/derecha/abajo ; None si no hay rostro o calibrando
    has_face: bool = False
    calibrating: bool = False
    calibration_progress: float = 0.0    # 0.0 - 1.0
    yaw: float = 0.0                     # grados relativos al "frente" calibrado
    pitch: float = 0.0
    raw_yaw: float = 0.0                 # grados crudos del modelo (sin calibrar)
    raw_pitch: float = 0.0
    face_bbox: Optional[Tuple[int, int, int, int]] = None   # (x1, y1, x2, y2) en píxeles


def _to_pixel_points(landmarks, width: int, height: int, normalized: bool) -> Optional[np.ndarray]:
    """Convierte lo que venga (lista MediaPipe, array Nx2/Nx3, lista de objetos .x/.y) a array (N,2) en píxeles."""
    if landmarks is None:
        return None
    try:
        if hasattr(landmarks, "landmark"):          # NormalizedLandmarkList de MediaPipe
            landmarks = landmarks.landmark
        if len(landmarks) == 0:
            return None
        if hasattr(landmarks[0], "x"):
            arr = np.array([[p.x, p.y] for p in landmarks], dtype=np.float64)
        else:
            arr = np.asarray(landmarks, dtype=np.float64)[:, :2].copy()
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < _MIN_LANDMARKS:
        return None
    if normalized:
        arr[:, 0] *= float(width)
        arr[:, 1] *= float(height)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


class HeadOrientationClassifier:
    """Uso:
        clf = HeadOrientationClassifier()
        pose = clf.classify(landmarks, frame_width, frame_height)   # una llamada por fotograma
        pose.label -> "frente" | "izquierda" | "derecha" | "abajo" | None
    """

    def __init__(self, config: Optional[HeadOrientationConfig] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = config or HeadOrientationConfig()
        self._clock = clock
        self._baseline: Optional[Tuple[float, float]] = None
        self._calib_samples: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.calibration_frames)
        self._calib_started_at: Optional[float] = None
        self._ema: Optional[Tuple[float, float]] = None
        self._last_label: Optional[str] = None
        self._last_result = HeadPoseResult()

    # ------------------------------------------------------------------ API pública
    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[Tuple[float, float]]:
        return self._baseline

    @property
    def last_result(self) -> HeadPoseResult:
        return self._last_result

    def reset_calibration(self) -> None:
        """Vuelve a calibrar (el conductor debe mirar al frente unos ~2 s)."""
        self._baseline = None
        self._calib_samples.clear()
        self._calib_started_at = None
        self._reset_tracking()

    def classify(self, landmarks, image_width: int, image_height: int,
                 normalized: bool = True) -> HeadPoseResult:
        pts = _to_pixel_points(landmarks, image_width, image_height, normalized)
        if pts is None:
            self._reset_tracking()
            self._last_result = HeadPoseResult(has_face=False)
            return self._last_result

        bbox = self._bbox(pts, image_width, image_height)
        est = self._estimate_pose(pts, image_width, image_height)
        if est is None:
            # rostro presente pero pose no confiable: se conserva la última etiqueta
            self._last_result = HeadPoseResult(label=self._last_label, has_face=True, face_bbox=bbox,
                                               calibrating=not self.is_calibrated)
            return self._last_result
        raw_yaw, raw_pitch = est

        if self._baseline is None:
            if self.cfg.auto_calibrate:
                self._collect_calibration(raw_yaw, raw_pitch)
            else:
                self._baseline = (0.0, 0.0)
        if self._baseline is None:
            progress = min(1.0, len(self._calib_samples) / float(self.cfg.calibration_frames))
            self._last_result = HeadPoseResult(label=None, has_face=True, calibrating=True,
                                               calibration_progress=progress, raw_yaw=raw_yaw,
                                               raw_pitch=raw_pitch, face_bbox=bbox)
            return self._last_result

        yaw = raw_yaw - self._baseline[0]
        pitch = raw_pitch - self._baseline[1]
        yaw, pitch = self._smooth(yaw, pitch)
        label = self._label_from(yaw, pitch)
        self._last_label = label
        self._last_result = HeadPoseResult(label=label, has_face=True, calibrating=False,
                                           calibration_progress=1.0, yaw=yaw, pitch=pitch,
                                           raw_yaw=raw_yaw, raw_pitch=raw_pitch, face_bbox=bbox)
        return self._last_result

    # ------------------------------------------------------------------ internos
    def _reset_tracking(self) -> None:
        self._ema = None
        self._last_label = None

    @staticmethod
    def _bbox(pts: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return (int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2)))

    def _estimate_pose(self, pts: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float]]:
        img_pts = np.ascontiguousarray(pts[list(_PNP_INDICES)], dtype=np.float64)
        eye_px = float(np.linalg.norm(img_pts[2] - img_pts[3]))
        if eye_px < self.cfg.min_eye_distance_px:
            return None
        f = float(w) * self.cfg.focal_length_factor
        cam = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Estimación inicial FRONTAL: evita que solvePnP caiga en la solución "espejada".
        z0 = f * _MODEL_EYE_DISTANCE / eye_px
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[(img_pts[0, 0] - w / 2.0) * z0 / f],
                         [(img_pts[0, 1] - h / 2.0) * z0 / f],
                         [z0]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, img_pts, cam, None, rvec, tvec, True,
                                          cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return None
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0:
            return None

        rot, _ = cv2.Rodrigues(rvec)
        gaze = rot @ np.array([0.0, 0.0, -1.0])   # hacia dónde apunta la cara (nariz hacia la cámara)
        yaw = math.degrees(math.atan2(gaze[0], -gaze[2]))
        pitch = math.degrees(math.atan2(gaze[1], math.hypot(gaze[0], gaze[2])))
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            return None
        return yaw, pitch

    def _collect_calibration(self, raw_yaw: float, raw_pitch: float) -> None:
        now = self._clock()
        if self._calib_started_at is None:
            self._calib_started_at = now
        c = self.cfg
        if abs(raw_yaw) <= c.calibration_max_abs_yaw and abs(raw_pitch) <= c.calibration_max_abs_pitch:
            self._calib_samples.append((raw_yaw, raw_pitch))
        if len(self._calib_samples) >= c.calibration_frames:
            arr = np.array(self._calib_samples)
            self._baseline = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
            logger.info("Calibracion de cabeza lista: yaw0=%.1f pitch0=%.1f", *self._baseline)
        elif now - self._calib_started_at > c.calibration_timeout_s:
            self._baseline = (0.0, 0.0)
            logger.warning("Calibracion de cabeza NO concluyo en %.0f s (el conductor no miro al frente); "
                           "se usa el modelo generico sin correccion.", c.calibration_timeout_s)

    def _smooth(self, yaw: float, pitch: float) -> Tuple[float, float]:
        a = self.cfg.smoothing
        if self._ema is None:
            self._ema = (yaw, pitch)
        else:
            self._ema = (a * yaw + (1 - a) * self._ema[0], a * pitch + (1 - a) * self._ema[1])
        return self._ema

    def _label_from(self, yaw: float, pitch: float) -> str:
        c = self.cfg
        prev = self._last_label
        yaw_thr = c.yaw_threshold * (c.hysteresis if prev in (IZQUIERDA, DERECHA) else 1.0)
        pit_thr = c.pitch_down_threshold * (c.hysteresis if prev == ABAJO else 1.0)
        yaw_excess = abs(yaw) / yaw_thr
        pitch_excess = pitch / pit_thr          # solo pitch > 0 (cabeza hacia abajo) cuenta
        if yaw_excess < 1.0 and pitch_excess < 1.0:
            return FRENTE
        if pitch_excess >= yaw_excess:
            return ABAJO
        turned_to_driver_left = yaw > 0
        if c.label_perspective == "screen":
            return DERECHA if turned_to_driver_left else IZQUIERDA
        return IZQUIERDA if turned_to_driver_left else DERECHA
