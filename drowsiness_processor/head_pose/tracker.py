"""
Cronómetro de distracción continua.

Mide cuántos segundos LLEVA el conductor distraído SIN interrupciones, usando el reloj real
(time.monotonic), por lo que no depende de cuántos FPS procese el servidor.

Tolerancia (grace_period_s): si la detección "parpadea" (p. ej. un fotograma en el que MediaPipe
pierde el rostro o la etiqueta cae a "frente" por ruido), el cronómetro NO se reinicia mientras la
atención no dure más de grace_period_s. Con grace_period_s=0 el reinicio es inmediato.
"""
from __future__ import annotations

import time
from typing import Callable, Optional


class DistractionTracker:
    def __init__(self, grace_period_s: float = 0.3, clock: Callable[[], float] = time.monotonic):
        if grace_period_s < 0:
            raise ValueError("grace_period_s no puede ser negativo")
        self.grace_period_s = float(grace_period_s)
        self._clock = clock
        self._start: Optional[float] = None
        self._last_distracted: Optional[float] = None

    def update(self, is_distracted: bool) -> float:
        """Llamar UNA vez por fotograma. Devuelve los segundos de distracción continua actuales."""
        now = self._clock()
        if is_distracted:
            if self._start is None:
                self._start = now
            self._last_distracted = now
        elif self._start is not None and self._last_distracted is not None:
            if now - self._last_distracted > self.grace_period_s:
                self.reset()
        return self.elapsed_s

    @property
    def elapsed_s(self) -> float:
        if self._start is None:
            return 0.0
        return max(0.0, self._clock() - self._start)

    @property
    def active(self) -> bool:
        return self._start is not None

    def reset(self) -> None:
        self._start = None
        self._last_distracted = None
