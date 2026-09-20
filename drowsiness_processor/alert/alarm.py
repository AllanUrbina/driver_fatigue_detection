"""
Alarma sonora NO bloqueante.

* request() solo compara tiempos y mete una tarea en una cola: tarda microsegundos, así que
  nunca detiene el procesamiento de video.
* El sonido se reproduce en UN hilo de fondo (compartido por todo el proceso, así que no se
  acumulan hilos aunque se creen varios sistemas/conexiones WebSocket).
* Cooldown: como máximo un pitido cada cooldown_s (1.2 s por defecto), aunque request() se
  llame en cada fotograma.
* Respaldo: si sounddevice / PortAudio no está disponible o no hay dispositivo de salida, se
  intenta winsound.Beep (Windows) y, si tampoco, un aviso por consola. Nunca lanza excepciones
  hacia el pipeline de video.

IMPORTANTE: el sonido sale por el equipo donde corre el backend (uvicorn/FastAPI). En la demo
backend y Flet corren en la misma PC, así que se escucha normalmente.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def make_beep(frequency_hz: float = 1000.0, duration_s: float = 0.25, volume: float = 0.6,
              sample_rate: int = 44100) -> np.ndarray:
    """Onda senoidal mono float32 con fade-in/out de 10 ms (evita el 'clic' al inicio y al final)."""
    n = max(1, int(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    wave = np.sin(2.0 * np.pi * frequency_hz * t)
    fade = min(int(sample_rate * 0.01), n // 2)
    if fade > 0:
        env = np.ones(n, dtype=np.float32)
        env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        wave = wave * env
    return (float(np.clip(volume, 0.0, 1.0)) * wave).astype(np.float32)


class _AudioWorker:
    """Un único hilo daemon por proceso que ejecuta las reproducciones una por una."""

    def __init__(self):
        self._queue: "queue.Queue[Callable[[], None]]" = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def submit(self, job: Callable[[], None]) -> bool:
        self._ensure_thread()
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:       # todavía hay un pitido pendiente: se descarta este
            return False

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="alarm-audio", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job()
            except Exception:                       # pragma: no cover - el audio nunca debe tumbar el proceso
                logger.exception("Fallo reproduciendo la alarma")


_WORKER = _AudioWorker()


class AlarmPlayer:
    def __init__(self, frequency_hz: float = 1000.0, beep_duration_s: float = 0.25, volume: float = 0.6,
                 cooldown_s: float = 1.2, sample_rate: int = 44100, device=None, enabled: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 play_fn: Optional[Callable[[np.ndarray, int], None]] = None):
        """play_fn(onda, sample_rate) permite inyectar otro backend (se usa en las pruebas)."""
        if cooldown_s < 0:
            raise ValueError("cooldown_s no puede ser negativo")
        self.cooldown_s = float(cooldown_s)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.enabled = enabled
        self._clock = clock
        self._custom_play_fn = play_fn
        self._wave = make_beep(frequency_hz, beep_duration_s, volume, self.sample_rate)
        self._frequency_hz = float(frequency_hz)
        self._duration_s = float(beep_duration_s)
        self._last_trigger: Optional[float] = None
        self._backend = "custom" if play_fn else "sounddevice"
        self._warned = False

    @property
    def backend(self) -> str:
        return self._backend

    def request(self) -> bool:
        """Pide un pitido. Devuelve True si se programó uno (False = en cooldown, deshabilitada o cola ocupada)."""
        if not self.enabled:
            return False
        now = self._clock()
        if self._last_trigger is not None and (now - self._last_trigger) < self.cooldown_s:
            return False
        if _WORKER.submit(self._play_once):
            self._last_trigger = now
            return True
        return False

    def reset(self) -> None:
        """Permite que el próximo request() suene de inmediato."""
        self._last_trigger = None

    # ------------------------------------------------------------------ reproducción (hilo de fondo)
    def _play_once(self) -> None:
        if self._custom_play_fn is not None:
            self._custom_play_fn(self._wave, self.sample_rate)
            return
        if self._backend == "sounddevice":
            try:
                import sounddevice as sd            # import perezoso: puede fallar si falta PortAudio (OSError)
                sd.play(self._wave, self.sample_rate, device=self.device)
                sd.wait()
                return
            except Exception as exc:                # PortAudioError, OSError, sin dispositivo, etc.
                self._warn(f"sounddevice no disponible ({exc!s}); se usa respaldo")
                self._backend = "winsound" if sys.platform.startswith("win") else "console"
        if self._backend == "winsound":
            try:
                import winsound
                winsound.Beep(int(self._frequency_hz), int(self._duration_s * 1000))
                return
            except Exception as exc:
                self._warn(f"winsound no disponible ({exc!s}); se usa la consola")
                self._backend = "console"
        sys.stdout.write("\a")
        sys.stdout.flush()

    def _warn(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("AlarmPlayer: %s", msg)
