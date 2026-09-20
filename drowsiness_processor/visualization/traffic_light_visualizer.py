"""
Dibuja el semáforo sobre el frame: círculo de color en la esquina superior izquierda con el
tiempo de distracción, más un panel con el estado, la orientación de la cabeza y el celular.

NOTA: cv2.putText (fuentes Hershey) no dibuja tildes ni ñ; por eso los textos van en mayúsculas
sin acentos ("DISTRAIDO", "POSIBLE LLAMADA"...).
"""
from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

from drowsiness_processor.alert.traffic_light import LightState, TrafficLightSnapshot

# Colores en BGR (formato de OpenCV)
COLORS = {
    LightState.GREEN: (0, 200, 0),
    LightState.YELLOW: (0, 220, 255),
    LightState.RED: (0, 0, 255),
    LightState.IDLE: (150, 150, 150),
}
STATE_TEXT = {
    LightState.GREEN: "ATENTO",
    LightState.YELLOW: "DISTRAIDO",
    LightState.RED: "ALERTA",
    LightState.IDLE: "ESPERA",
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_size(text: str, scale: float, thickness: int) -> Tuple[int, int]:
    (w, h), base = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h + base


def _put_outlined(img, text, org, scale, color, thickness=1):
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


class TrafficLightVisualizer:
    def __init__(self, radius: int = 0, margin: int = 14, anchor: str = "top_left",
                 show_details: bool = True, show_debug: bool = False, red_border: bool = True):
        """radius=0 -> se calcula según el tamaño del frame. anchor: 'top_left' | 'top_right'."""
        if anchor not in ("top_left", "top_right"):
            raise ValueError("anchor debe ser 'top_left' o 'top_right'")
        self.radius = radius
        self.margin = margin
        self.anchor = anchor
        self.show_details = show_details
        self.show_debug = show_debug
        self.red_border = red_border

    def draw(self, frame: np.ndarray, snap: TrafficLightSnapshot) -> np.ndarray:
        """Dibuja EN SITIO sobre 'frame' (BGR uint8) y lo devuelve."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        r = self.radius or max(24, int(min(h, w) * 0.075))
        cx = self.margin + r if self.anchor == "top_left" else w - self.margin - r
        cy = self.margin + r
        color = COLORS[snap.state]

        if snap.state == LightState.RED and self.red_border:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, max(4, r // 6))

        # --- panel de texto (semitransparente) junto al círculo
        lines, line_colors = self._detail_lines(snap)
        if self.show_details and lines:
            scale = max(0.45, r / 62.0)
            th = 1
            sizes = [_text_size(t, scale, th) for t in lines]
            pw = max(s[0] for s in sizes) + 16
            ph = sum(s[1] + 6 for s in sizes) + 8
            if self.anchor == "top_left":
                px1 = cx + r + 8
            else:
                px1 = cx - r - 8 - pw
            px1 = int(np.clip(px1, 0, max(0, w - pw)))
            py1 = cy - r
            px2, py2 = min(w, px1 + pw), min(h, py1 + ph)
            if px2 > px1 and py2 > py1:
                roi = frame[py1:py2, px1:px2]
                frame[py1:py2, px1:px2] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
            y = py1 + 6
            for text, col, (tw, thh) in zip(lines, line_colors, sizes):
                y += thh
                _put_outlined(frame, text, (px1 + 8, y), scale, col, th)
                y += 6

        # --- círculo del semáforo
        cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)
        if snap.state == LightState.RED and int(time.monotonic() * 4) % 2 == 0:
            cv2.circle(frame, (cx, cy), r + 6, color, 3, cv2.LINE_AA)   # anillo que parpadea en ROJO

        # --- tiempo transcurrido dentro del círculo
        if snap.state == LightState.IDLE:
            txt = f"{int(snap.calibration_progress * 100)}%" if snap.calibrating else "--"
        else:
            txt = ">99s" if snap.elapsed_s >= 100 else f"{snap.elapsed_s:.1f}s"
        scale = max(0.4, r / 58.0)
        tw, tth = _text_size(txt, scale, 2)
        _put_outlined(frame, txt, (cx - tw // 2, cy + tth // 3), scale, (255, 255, 255), 2)
        return frame

    def _detail_lines(self, snap: TrafficLightSnapshot):
        color = COLORS[snap.state]
        if snap.state == LightState.IDLE and snap.calibrating:
            title = "CALIBRANDO: MIRA AL FRENTE"
        else:
            title = STATE_TEXT[snap.state]
        lines, cols = [title], [color]
        head = snap.head_label if snap.head_label else ("--" if snap.calibrating or snap.reason == "espera" else "sin rostro")
        lines.append(f"Cabeza: {head}")
        cols.append((255, 255, 255))
        if snap.possible_call:
            phone_txt, phone_col = "POSIBLE LLAMADA", (0, 0, 255)
        elif snap.phone_visible:
            phone_txt, phone_col = "Celular: SI", (0, 220, 255)
        else:
            phone_txt, phone_col = "Celular: no", (255, 255, 255)
        lines.append(phone_txt)
        cols.append(phone_col)
        if self.show_debug:
            lines.append(f"yaw {snap.yaw:+.0f}  pitch {snap.pitch:+.0f}")
            cols.append((200, 200, 200))
        return lines, cols
