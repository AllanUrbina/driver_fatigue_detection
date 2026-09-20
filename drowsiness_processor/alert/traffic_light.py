"""
Semáforo de atención (VERDE / AMARILLO / ROJO).

Reglas (todas configurables en TrafficLightConfig):
  VERDE    : conductor atento al frente y sin celular (o distraído menos de yellow_after_s).
  AMARILLO : distracción continua entre yellow_after_s (1.0 s) y red_after_s (2.5 s).
  ROJO     : distracción continua mayor a red_after_s (2.5 s), o "posible llamada"
             (celular cerca de la cara) -> rojo directo.
  ESPERA   : gris. Aún no se ha visto al conductor o se está calibrando la cabeza.

"Distracción" = cabeza fuera de "frente"  O  celular visible  O  rostro perdido (configurable).
El tiempo se cuenta desde que empieza la distracción continua (DistractionTracker).
Cuando el estado es ROJO se pide un pitido a AlarmPlayer (que aplica su propio cooldown).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from drowsiness_processor.alert.alarm import AlarmPlayer
from drowsiness_processor.head_pose.tracker import DistractionTracker

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) en píxeles


class LightState(str, Enum):
    GREEN = "verde"
    YELLOW = "amarillo"
    RED = "rojo"
    IDLE = "espera"


@dataclass
class TrafficLightConfig:
    yellow_after_s: float = 1.0                 # a partir de aquí: AMARILLO
    red_after_s: float = 2.5                    # por encima de aquí: ROJO
    grace_period_s: float = 0.3                 # tolerancia a parpadeos de la detección
    phone_counts_as_distraction: bool = True    # celular visible = distracción (VERDE exige "sin celular")
    no_face_is_distraction: bool = True         # rostro perdido (p. ej. giro extremo) = distracción
    require_face_first: bool = True             # el semáforo espera a ver al conductor al menos una vez
    possible_call_confirm_frames: int = 2       # fotogramas seguidos con celular junto a la cara -> ROJO directo
    phone_face_margin_x: float = 0.30           # "cerca de la cara" = solapa con la caja del rostro ampliada
    phone_face_margin_y: float = 0.20           #   30 % del ancho a cada lado y 20 % del alto arriba/abajo
    face_memory_s: float = 1.0                  # si se pierde el rostro, se usa su última caja este tiempo
    alarm_enabled: bool = True

    def __post_init__(self):
        if self.yellow_after_s < 0 or self.red_after_s <= self.yellow_after_s:
            raise ValueError("Se requiere 0 <= yellow_after_s < red_after_s")
        if self.possible_call_confirm_frames < 1:
            raise ValueError("possible_call_confirm_frames debe ser >= 1")


@dataclass
class TrafficLightSnapshot:
    state: LightState = LightState.IDLE
    elapsed_s: float = 0.0
    reason: str = "espera"
    head_label: Optional[str] = None
    phone_visible: bool = False
    possible_call: bool = False
    calibrating: bool = False
    calibration_progress: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    alarm_requested: bool = False

    def to_dict(self) -> dict:
        """Solo tipos nativos de Python (serializable por FastAPI/JSON)."""
        return {
            "state": self.state.value,
            "elapsed_s": round(float(self.elapsed_s), 2),
            "reason": self.reason,
            "head_orientation": self.head_label,
            "phone_visible": bool(self.phone_visible),
            "possible_call": bool(self.possible_call),
            "calibrating": bool(self.calibrating),
            "yaw": round(float(self.yaw), 1),
            "pitch": round(float(self.pitch), 1),
            "alarm_requested": bool(self.alarm_requested),
        }


# ---------------------------------------------------------------------- utilidades de celular
def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "cpu"):                  # tensor de torch
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def extract_phone_boxes(detections, class_ids: Optional[Sequence[int]] = None,
                        min_conf: float = 0.0) -> List[Box]:
    """Normaliza la salida de CUALQUIER detector de celular a una lista de cajas (x1, y1, x2, y2).

    Acepta: None, resultados de ultralytics (uno o lista), array Nx4/Nx5/Nx6 (x1,y1,x2,y2[,conf[,cls]]),
    lista de tuplas/listas, o lista de dicts con 'bbox' | 'box' | 'xyxy' (y opcional 'conf'/'cls').
    class_ids: si se indica, solo se conservan esas clases (en COCO 'cell phone' = 67). Si tu modelo ya
    devuelve únicamente celulares, déjalo en None.
    Nunca lanza excepciones: ante un formato desconocido devuelve [] y avisa una vez en el log.
    """
    if detections is None:
        return []
    keep = set(int(c) for c in class_ids) if class_ids is not None else None
    boxes: List[Box] = []
    try:
        items = detections if isinstance(detections, (list, tuple)) else [detections]
        for item in items:
            if item is None:
                continue
            if hasattr(item, "boxes") and hasattr(item.boxes, "xyxy"):          # ultralytics Results
                xyxy = _to_numpy(item.boxes.xyxy).reshape(-1, 4)
                cls = _to_numpy(item.boxes.cls).reshape(-1) if getattr(item.boxes, "cls", None) is not None else None
                conf = _to_numpy(item.boxes.conf).reshape(-1) if getattr(item.boxes, "conf", None) is not None else None
                for i, row in enumerate(xyxy):
                    if keep is not None and cls is not None and int(cls[i]) not in keep:
                        continue
                    if conf is not None and float(conf[i]) < min_conf:
                        continue
                    boxes.append(tuple(float(v) for v in row))
                continue
            if isinstance(item, dict):
                raw = item.get("bbox", item.get("box", item.get("xyxy")))
                if raw is None:
                    continue
                if keep is not None and item.get("cls", item.get("class_id")) is not None \
                        and int(item.get("cls", item.get("class_id"))) not in keep:
                    continue
                if item.get("conf", item.get("confidence")) is not None \
                        and float(item.get("conf", item.get("confidence"))) < min_conf:
                    continue
                row = _to_numpy(raw).reshape(-1)
                boxes.append(tuple(float(v) for v in row[:4]))
                continue
            arr = _to_numpy(item)
            if arr.ndim == 1 and arr.size >= 4:                                # una sola caja
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 4:                            # varias cajas
                for row in arr:
                    if arr.shape[1] >= 5 and float(row[4]) < min_conf:
                        continue
                    if keep is not None and arr.shape[1] >= 6 and int(row[5]) not in keep:
                        continue
                    boxes.append(tuple(float(v) for v in row[:4]))
    except Exception as exc:
        logger.warning("extract_phone_boxes: formato de deteccion no reconocido (%s)", exc)
        return []
    out: List[Box] = []
    for x1, y1, x2, y2 in boxes:
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def phone_near_face(phone_box: Box, face_bbox: Tuple[float, float, float, float],
                    margin_x: float = 0.30, margin_y: float = 0.20) -> bool:
    """True si la caja del celular se solapa con la caja del rostro ampliada (zona de la oreja/mejilla/boca)."""
    fx1, fy1, fx2, fy2 = face_bbox
    fw, fh = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
    ex1, ex2 = fx1 - margin_x * fw, fx2 + margin_x * fw
    ey1, ey2 = fy1 - margin_y * fh, fy2 + margin_y * fh
    px1, py1, px2, py2 = phone_box
    return not (px2 < ex1 or px1 > ex2 or py2 < ey1 or py1 > ey2)


# ---------------------------------------------------------------------- semáforo
class TrafficLightSystem:
    def __init__(self, config: Optional[TrafficLightConfig] = None, alarm: Optional[AlarmPlayer] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = config or TrafficLightConfig()
        self.alarm = alarm
        self._clock = clock
        self.tracker = DistractionTracker(self.cfg.grace_period_s, clock=clock)
        self._seen_face = False
        self._near_frames = 0
        self._last_face_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_face_time: Optional[float] = None
        self.last_snapshot = TrafficLightSnapshot()

    def reset(self) -> None:
        self.tracker.reset()
        self._seen_face = False
        self._near_frames = 0
        self._last_face_bbox = None
        self._last_face_time = None
        if self.alarm is not None:
            self.alarm.reset()

    def update(self, pose, phone_boxes: Optional[Sequence[Box]] = None) -> TrafficLightSnapshot:
        """pose: HeadPoseResult del clasificador. phone_boxes: salida de extract_phone_boxes()."""
        cfg = self.cfg
        now = self._clock()
        boxes = list(phone_boxes) if phone_boxes else []
        phone_visible = len(boxes) > 0

        has_face = bool(pose.has_face)
        if has_face:
            self._seen_face = True
            if pose.face_bbox is not None:
                self._last_face_bbox, self._last_face_time = pose.face_bbox, now

        # ¿celular junto a la cara? (si se acaba de perder el rostro se usa su última caja conocida)
        ref_bbox = None
        if has_face and pose.face_bbox is not None:
            ref_bbox = pose.face_bbox
        elif self._last_face_bbox is not None and self._last_face_time is not None \
                and (now - self._last_face_time) <= cfg.face_memory_s:
            ref_bbox = self._last_face_bbox
        near = bool(ref_bbox is not None and any(
            phone_near_face(b, ref_bbox, cfg.phone_face_margin_x, cfg.phone_face_margin_y) for b in boxes))
        self._near_frames = self._near_frames + 1 if near else 0
        possible_call = self._near_frames >= cfg.possible_call_confirm_frames

        armed = self._seen_face or not cfg.require_face_first
        if not armed:                                   # nadie frente a la cámara todavía
            self.tracker.reset()
            snap = TrafficLightSnapshot(state=LightState.IDLE, reason="espera", phone_visible=phone_visible)
            self.last_snapshot = snap
            return snap

        head_label = pose.label if (has_face and not pose.calibrating) else None
        head_distracted = head_label not in (None, "frente")
        no_face_distracted = (not has_face) and cfg.no_face_is_distraction
        phone_distracted = phone_visible and cfg.phone_counts_as_distraction
        is_distracted = head_distracted or no_face_distracted or phone_distracted or possible_call

        elapsed = self.tracker.update(is_distracted)

        if possible_call:
            state, reason = LightState.RED, "posible llamada"
        else:
            if elapsed > cfg.red_after_s:
                state = LightState.RED
            elif elapsed >= cfg.yellow_after_s:
                state = LightState.YELLOW
            else:
                state = LightState.GREEN
            if phone_distracted:
                reason = "celular"
            elif no_face_distracted:
                reason = "sin rostro"
            elif head_distracted:
                reason = f"cabeza {head_label}"
            elif pose.calibrating:
                reason = "calibrando"
            else:
                reason = "atento"

        calibrating = bool(has_face and pose.calibrating)
        if calibrating and state == LightState.GREEN and not is_distracted and elapsed == 0.0:
            state = LightState.IDLE                      # gris mientras calibra

        alarm_requested = False
        if state == LightState.RED and cfg.alarm_enabled and self.alarm is not None:
            alarm_requested = bool(self.alarm.request())

        snap = TrafficLightSnapshot(
            state=state, elapsed_s=elapsed, reason=reason, head_label=head_label,
            phone_visible=phone_visible, possible_call=possible_call, calibrating=calibrating,
            calibration_progress=float(pose.calibration_progress), yaw=float(pose.yaw),
            pitch=float(pose.pitch), alarm_requested=alarm_requested)
        self.last_snapshot = snap
        return snap
