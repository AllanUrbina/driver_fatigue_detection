"""
VERIFICACION PREVIA A LA DEMO - orientacion de cabeza + semaforo + alarma.

Ejecutar SIEMPRE desde la raiz del proyecto:

    python tests/test_distraction_system.py              # todas las pruebas automaticas
    python tests/test_distraction_system.py --mute       # igual, pero sin sonido real
    python tests/test_distraction_system.py --beep       # solo prueba el sonido (3 pitidos)
    python tests/test_distraction_system.py --camera 1   # ensayo en vivo con tu webcam (indice 1)

Codigo de salida: 0 = todo OK, 1 = hay fallos (no hagas la demo hasta corregirlos).
"""
import argparse
import base64
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # el proyecto usa rutas relativas (reports/...), asi que hay que estar en la raiz

try:  # evita UnicodeEncodeError en consolas de Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

# ----------------------------------------------------------------------------- mini framework
TESTS = []


class Skip(Exception):
    pass


class Warn(Exception):
    pass


def test(section):
    def deco(fn):
        TESTS.append((section, fn.__name__[5:].replace("_", " "), fn))
        return fn
    return deco


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def wait_until(cond, timeout=2.0, step=0.01):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


# ----------------------------------------------------------------------------- datos sinteticos
# Geometria 3D (px, imagen 480x720) de los 6 landmarks usados, tomada de una cara real frontal
# procesada con MediaPipe. Sirve para simular giros de cabeza con angulos CONOCIDOS.
REF_IDX = [1, 152, 33, 263, 61, 291]
REF_3D = np.array([[267.0, 421.4, -89.2], [257.0, 604.0, 23.4], [127.5, 314.0, 15.4],
                   [371.9, 315.7, 32.0], [184.5, 495.7, 6.2], [325.7, 491.9, 16.8]])
W, H = 480, 720


def synth_landmarks(yaw_deg, pitch_deg, rng=None, noise=0.0, eye_noise=0.0):
    """yaw>0: la nariz se mueve hacia la derecha de la imagen. pitch>0: la cabeza baja."""
    a, b = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    c = REF_3D.mean(axis=0) + np.array([0, 0, 60.0])
    p = (rx @ ry @ (REF_3D - c).T).T + c
    z = 480.0 + (p[:, 2] - c[2])
    xy = np.stack([W / 2 + W * (p[:, 0] - c[0]) / z, H / 2 + W * (p[:, 1] - c[1]) / z], axis=1)
    if rng is not None and noise:
        xy = xy + rng.normal(0, noise, xy.shape)
    if rng is not None and eye_noise:
        xy[2:4] += rng.normal(0, eye_noise, (2, 2))
    lm = np.zeros((478, 3))
    lm[:, :2] = xy.mean(axis=0) / np.array([W, H])
    lm[REF_IDX, :2] = xy / np.array([W, H])
    return lm


def calibrated_classifier(**cfg_kwargs):
    from drowsiness_processor.head_pose.orientation import HeadOrientationClassifier, HeadOrientationConfig
    clf = HeadOrientationClassifier(HeadOrientationConfig(**cfg_kwargs))
    rng = np.random.default_rng(3)
    for _ in range(clf.cfg.calibration_frames + 2):
        clf.classify(synth_landmarks(0, 0, rng, 0.8), W, H)
    assert clf.is_calibrated, "no se calibro con fotogramas frontales"
    return clf


def pose(label="frente", has_face=True, calibrating=False, bbox=(200, 100, 440, 420)):
    from drowsiness_processor.head_pose.orientation import HeadPoseResult
    return HeadPoseResult(label=None if calibrating else label, has_face=has_face, calibrating=calibrating,
                          calibration_progress=0.5 if calibrating else 1.0,
                          face_bbox=bbox if has_face else None)


class FakeAlarm:
    def __init__(self):
        self.calls = 0

    def request(self):
        self.calls += 1
        return True

    def reset(self):
        pass


def make_light(**kw):
    from drowsiness_processor.alert.traffic_light import TrafficLightSystem, TrafficLightConfig
    clock = FakeClock()
    alarm = FakeAlarm()
    return TrafficLightSystem(TrafficLightConfig(**kw), alarm=alarm, clock=clock), clock, alarm


# =============================================================================== 1. MODULOS
S1 = "1. Modulos y dependencias"


@test(S1)
def test_importa_los_modulos_nuevos():
    import importlib
    for m in ("drowsiness_processor.head_pose.orientation", "drowsiness_processor.head_pose.tracker",
              "drowsiness_processor.alert.traffic_light", "drowsiness_processor.alert.alarm",
              "drowsiness_processor.visualization.traffic_light_visualizer"):
        importlib.import_module(m)


@test(S1)
def test_hook_last_landmarks_en_facemesh():
    from drowsiness_processor.extract_points.face_mesh.face_mesh_processor import FaceMeshProcessor
    fm = FaceMeshProcessor()
    assert hasattr(fm, "last_landmarks"), (
        "FALTA el hook: agrega 'self.last_landmarks = None' y el guardado de landmarks en "
        "extract_points/face_mesh/face_mesh_processor.py (ver instrucciones, paso 2)")


@test(S1)
def test_sounddevice_y_dispositivo_de_salida():
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="output")
    except Exception as exc:
        raise Warn(f"sounddevice no puede usarse ({exc}). Hay respaldo (winsound/consola) pero revisa el audio")
    if not dev or dev.get("max_output_channels", 0) < 1:
        raise Warn("no hay dispositivo de salida de audio por defecto")


# =============================================================================== 2. TRACKER
S2 = "2. Cronometro de distraccion"


@test(S2)
def test_mide_la_distraccion_continua():
    from drowsiness_processor.head_pose.tracker import DistractionTracker
    clk = FakeClock()
    tr = DistractionTracker(0.3, clock=clk)
    assert tr.update(False) == 0.0
    tr.update(True)
    clk.advance(1.0)
    assert abs(tr.update(True) - 1.0) < 1e-9
    clk.advance(1.5)
    assert abs(tr.update(True) - 2.5) < 1e-9


@test(S2)
def test_se_reinicia_al_recuperar_atencion():
    from drowsiness_processor.head_pose.tracker import DistractionTracker
    clk = FakeClock()
    tr = DistractionTracker(0.3, clock=clk)
    tr.update(True)
    clk.advance(1.0)
    tr.update(True)
    clk.advance(0.5)
    assert tr.update(False) == 0.0 and not tr.active


@test(S2)
def test_tolera_parpadeos_cortos_de_la_deteccion():
    from drowsiness_processor.head_pose.tracker import DistractionTracker
    clk = FakeClock()
    tr = DistractionTracker(0.3, clock=clk)
    tr.update(True)
    clk.advance(1.0)
    tr.update(True)
    clk.advance(0.2)                       # un instante "atento" < 0.3 s: NO reinicia
    tr.update(False)
    clk.advance(0.1)
    assert abs(tr.update(True) - 1.3) < 1e-9


# =============================================================================== 3. SEMAFORO
S3 = "3. Semaforo (logica)"


@test(S3)
def test_verde_si_esta_atento_y_sin_celular():
    from drowsiness_processor.alert.traffic_light import LightState
    tl, clk, _ = make_light()
    for _ in range(20):
        s = tl.update(pose("frente"), [])
        clk.advance(0.1)
    assert s.state == LightState.GREEN and s.elapsed_s == 0.0 and s.reason == "atento"


@test(S3)
def test_umbrales_verde_amarillo_rojo():
    from drowsiness_processor.alert.traffic_light import LightState as L
    tl, clk, _ = make_light()                       # 1.0 s y 2.5 s
    tl.update(pose("frente"), [])
    esperado = [(0.0, L.GREEN), (0.5, L.GREEN), (0.99, L.GREEN), (1.0, L.YELLOW), (1.5, L.YELLOW),
                (2.49, L.YELLOW), (2.51, L.RED), (4.0, L.RED)]
    t0 = clk.t
    for t, estado in esperado:
        clk.t = t0 + 10 + t
        if t == 0.0:
            tl.tracker.reset()
        s = tl.update(pose("izquierda"), [])
        assert s.state == estado, f"a los {t}s se esperaba {estado.value} y salio {s.state.value}"


@test(S3)
def test_el_tiempo_cuenta_desde_el_inicio_de_la_distraccion():
    tl, clk, _ = make_light()
    for _ in range(10):                              # 1 s atento
        tl.update(pose("frente"), [])
        clk.advance(0.1)
    tl.update(pose("derecha"), [])                   # empieza la distraccion
    clk.advance(1.2)
    s = tl.update(pose("derecha"), [])
    assert abs(s.elapsed_s - 1.2) < 1e-6 and s.state.value == "amarillo"


@test(S3)
def test_vuelve_a_verde_al_recuperar_atencion():
    tl, clk, _ = make_light()
    tl.update(pose("frente"), [])
    tl.update(pose("abajo"), [])
    clk.advance(3.0)
    assert tl.update(pose("abajo"), []).state.value == "rojo"
    clk.advance(0.1)
    tl.update(pose("frente"), [])
    clk.advance(0.5)
    s = tl.update(pose("frente"), [])
    assert s.state.value == "verde" and s.elapsed_s == 0.0


@test(S3)
def test_posible_llamada_es_rojo_directo():
    tl, clk, alarm = make_light()
    tl.update(pose("frente"), [])
    cerca = [(430, 200, 520, 380)]                  # pegado al costado de la cara (bbox 200..440)
    s1 = tl.update(pose("frente"), cerca)
    clk.advance(0.1)
    s2 = tl.update(pose("frente"), cerca)
    assert s2.possible_call and s2.state.value == "rojo" and s2.elapsed_s < 0.5, s2
    assert s2.reason == "posible llamada" and alarm.calls >= 1
    assert not s1.possible_call, "con possible_call_confirm_frames=2 el primer fotograma aun no confirma"


@test(S3)
def test_celular_lejos_de_la_cara_cuenta_como_distraccion_temporizada():
    tl, clk, _ = make_light()
    tl.update(pose("frente"), [])
    lejos = [(20, 600, 120, 700)]
    estados = []
    for _ in range(30):
        s = tl.update(pose("frente"), lejos)
        estados.append(s.state.value)
        clk.advance(0.1)
    assert not s.possible_call and s.phone_visible
    assert "amarillo" in estados and estados[-1] == "rojo" and estados[0] == "verde", estados


@test(S3)
def test_sin_rostro_espera_al_conductor_y_luego_alarma():
    tl, clk, _ = make_light()
    s = tl.update(pose(has_face=False), [])
    assert s.state.value == "espera"                # nadie ha aparecido aun: no hay alarma
    tl.update(pose("frente"), [])
    for _ in range(30):
        clk.advance(0.1)
        s = tl.update(pose(has_face=False), [])
    assert s.state.value == "rojo" and s.reason == "sin rostro"


@test(S3)
def test_calibrando_no_cuenta_como_distraccion():
    tl, clk, _ = make_light()
    for _ in range(40):
        s = tl.update(pose(calibrating=True), [])
        clk.advance(0.1)
    assert s.state.value == "espera" and s.calibrating and s.elapsed_s == 0.0


@test(S3)
def test_la_alarma_se_pide_solo_en_rojo():
    tl, clk, alarm = make_light()
    tl.update(pose("frente"), [])
    for _ in range(20):                              # 2 s de distraccion: amarillo, aun sin alarma
        tl.update(pose("izquierda"), [])
        clk.advance(0.1)
    assert alarm.calls == 0
    for _ in range(10):
        tl.update(pose("izquierda"), [])
        clk.advance(0.1)
    assert alarm.calls > 0


@test(S3)
def test_umbrales_configurables_y_validacion():
    from drowsiness_processor.alert.traffic_light import TrafficLightConfig
    tl, clk, _ = make_light(yellow_after_s=0.5, red_after_s=1.0)
    tl.update(pose("frente"), [])
    tl.update(pose("izquierda"), [])
    clk.advance(0.6)
    assert tl.update(pose("izquierda"), []).state.value == "amarillo"
    clk.advance(0.6)
    assert tl.update(pose("izquierda"), []).state.value == "rojo"
    try:
        TrafficLightConfig(yellow_after_s=3.0, red_after_s=2.0)
    except ValueError:
        return
    raise AssertionError("debio rechazar red_after_s <= yellow_after_s")


@test(S3)
def test_snapshot_es_serializable_a_json():
    tl, clk, _ = make_light()
    p = pose("frente")
    p.yaw, p.pitch = np.float32(3.14159), np.float64(-2.5)   # tipos numpy: FastAPI/JSON los rechazaria
    s = tl.update(p, [])
    json.dumps(s.to_dict())


@test(S3)
def test_extract_phone_boxes_acepta_varios_formatos():
    from drowsiness_processor.alert.traffic_light import extract_phone_boxes

    class B:  # imita ultralytics.engine.results.Boxes
        xyxy = np.array([[10, 20, 110, 220], [0, 0, 50, 50]], dtype=float)
        cls = np.array([67, 0])
        conf = np.array([0.9, 0.8])

    class R:
        boxes = B()

    assert extract_phone_boxes(None) == [] and extract_phone_boxes([]) == []
    assert extract_phone_boxes([(1, 2, 30, 40)]) == [(1.0, 2.0, 30.0, 40.0)]
    assert extract_phone_boxes([{"bbox": [1, 2, 30, 40], "conf": 0.9}]) == [(1.0, 2.0, 30.0, 40.0)]
    assert extract_phone_boxes(np.array([[1, 2, 30, 40, 0.9, 67]])) == [(1.0, 2.0, 30.0, 40.0)]
    assert len(extract_phone_boxes([R()])) == 2
    assert extract_phone_boxes([R()], class_ids=(67,)) == [(10.0, 20.0, 110.0, 220.0)]
    assert extract_phone_boxes("basura") == [] and extract_phone_boxes([(5, 5, 5, 5)]) == []


# =============================================================================== 4. ORIENTACION
S4 = "4. Orientacion de cabeza"


@test(S4)
def test_clasifica_frente_izquierda_derecha_abajo():
    casos = [(0, 0, "frente"), (10, 0, "frente"), (-10, 0, "frente"), (0, 10, "frente"), (0, -15, "frente"),
             (45, 0, "izquierda"), (60, 0, "izquierda"), (-45, 0, "derecha"), (-60, 0, "derecha"),
             (0, 35, "abajo"), (0, 45, "abajo"), (5, 40, "abajo")]
    fallos = []
    for yaw, pitch, esperado in casos:
        clf = calibrated_classifier()
        rng = np.random.default_rng(11)
        for _ in range(8):
            r = clf.classify(synth_landmarks(yaw, pitch, rng, 1.0), W, H)
        if r.label != esperado:
            fallos.append(f"(yaw={yaw}, pitch={pitch}) esperado={esperado} obtenido={r.label} "
                          f"[yaw_est={r.yaw:.1f} pitch_est={r.pitch:.1f}]")
    assert not fallos, "\n      ".join(fallos)


@test(S4)
def test_convencion_de_signos_conductor_vs_pantalla():
    clf = calibrated_classifier(label_perspective="driver")
    for _ in range(6):
        r = clf.classify(synth_landmarks(50, 0), W, H)   # nariz hacia la derecha de la imagen
    assert r.label == "izquierda" and r.yaw > 0, "la nariz a la derecha de la imagen = giro a la IZQUIERDA del conductor"
    clf = calibrated_classifier(label_perspective="screen")
    for _ in range(6):
        r = clf.classify(synth_landmarks(50, 0), W, H)
    assert r.label == "derecha"


@test(S4)
def test_estable_frente_al_ruido_de_landmarks():
    clf = calibrated_classifier()
    rng = np.random.default_rng(5)
    etiquetas = [clf.classify(synth_landmarks(0, 0, rng, 1.5), W, H).label for _ in range(200)]
    assert set(etiquetas) == {"frente"}, f"parpadeo de etiqueta: {set(etiquetas)}"


@test(S4)
def test_tolera_lentes_ruido_extra_en_los_ojos():
    clf = calibrated_classifier()
    rng = np.random.default_rng(9)
    etiquetas = [clf.classify(synth_landmarks(0, 0, rng, 1.0, eye_noise=4.0), W, H).label for _ in range(200)]
    frac = etiquetas.count("frente") / len(etiquetas)
    assert frac >= 0.95, f"solo {frac:.0%} 'frente' con ojos ruidosos"
    etiquetas = [clf.classify(synth_landmarks(45, 0, rng, 1.0, eye_noise=4.0), W, H).label for _ in range(100)]
    assert etiquetas.count("izquierda") / len(etiquetas) >= 0.9


@test(S4)
def test_histeresis_evita_parpadeo_en_el_umbral():
    clf = calibrated_classifier()
    rng = np.random.default_rng(2)
    secuencia = list(np.linspace(0, 50, 40)) + list(np.linspace(50, 0, 40))
    etiquetas = [clf.classify(synth_landmarks(y, 0, rng, 0.7), W, H).label for y in secuencia]
    cambios = sum(1 for a, b in zip(etiquetas, etiquetas[1:]) if a != b)
    assert cambios <= 2, f"{cambios} cambios de etiqueta en una rampa ida y vuelta (esperado <= 2): {etiquetas}"


@test(S4)
def test_calibracion_automatica_y_recalibracion():
    from drowsiness_processor.head_pose.orientation import HeadOrientationClassifier
    clf = HeadOrientationClassifier()
    r = clf.classify(synth_landmarks(0, 0), W, H)
    assert r.calibrating and r.label is None and 0 < r.calibration_progress < 1
    for _ in range(20):
        r = clf.classify(synth_landmarks(0, 0), W, H)
    assert clf.is_calibrated and r.label == "frente"
    clf.reset_calibration()
    assert not clf.is_calibrated and clf.classify(synth_landmarks(0, 0), W, H).calibrating


@test(S4)
def test_sin_rostro_o_datos_invalidos_no_lanza_excepciones():
    from drowsiness_processor.head_pose.orientation import HeadOrientationClassifier
    clf = HeadOrientationClassifier()
    for basura in (None, [], np.zeros((10, 3)), "hola", [[1, 2]], np.full((478, 3), np.nan), 5):
        r = clf.classify(basura, W, H)
        assert not r.has_face and r.label is None


# =============================================================================== 5. ALARMA
S5 = "5. Alarma sonora"


@test(S5)
def test_onda_del_pitido():
    from drowsiness_processor.alert.alarm import make_beep
    w = make_beep(1000, 0.25, 0.6, 44100)
    assert w.dtype == np.float32 and w.ndim == 1 and len(w) == int(44100 * 0.25)
    assert 0.55 < float(np.abs(w).max()) <= 0.6 and abs(float(w[0])) < 0.01 and abs(float(w[-1])) < 0.01


@test(S5)
def test_cooldown_un_pitido_cada_1_2_segundos():
    from drowsiness_processor.alert.alarm import AlarmPlayer
    clk = FakeClock()
    sonados = []
    a = AlarmPlayer(cooldown_s=1.2, clock=clk, play_fn=lambda w, sr: sonados.append(1))
    aceptados = 0
    for _ in range(24):                              # 24 fotogramas en 1.15 s (< cooldown)
        aceptados += a.request()
        clk.advance(0.05)
    assert aceptados == 1
    assert wait_until(lambda: len(sonados) == 1), "el primer pitido no sono"
    clk.advance(0.2)                                 # ya pasaron > 1.2 s desde el primero
    assert a.request() is True
    assert wait_until(lambda: len(sonados) == 2)
    assert a.request() is False                      # de nuevo en cooldown


@test(S5)
def test_no_bloquea_el_hilo_principal():
    from drowsiness_processor.alert.alarm import AlarmPlayer
    lento = lambda w, sr: time.sleep(0.6)            # backend de audio muy lento
    a = AlarmPlayer(cooldown_s=0.0, play_fn=lento)
    t0 = time.perf_counter()
    for _ in range(200):
        a.request()
    dt = time.perf_counter() - t0
    assert dt < 0.05, f"request() bloqueo {dt * 1000:.0f} ms para 200 llamadas"
    wait_until(lambda: False, timeout=0.7)           # deja terminar el pitido lento


@test(S5)
def test_un_solo_hilo_de_audio_aunque_haya_muchos_sistemas():
    from drowsiness_processor.alert.alarm import AlarmPlayer
    for _ in range(20):
        AlarmPlayer(play_fn=lambda w, sr: None).request()
    time.sleep(0.2)
    hilos = [t for t in threading.enumerate() if t.name == "alarm-audio"]
    assert len(hilos) <= 1, f"{len(hilos)} hilos de audio"


@test(S5)
def test_deshabilitada_no_suena():
    from drowsiness_processor.alert.alarm import AlarmPlayer
    a = AlarmPlayer(enabled=False, play_fn=lambda w, sr: (_ for _ in ()).throw(AssertionError("sono")))
    assert a.request() is False


# =============================================================================== 6. VISUALIZADOR
S6 = "6. Dibujo del semaforo"


@test(S6)
def test_circulo_de_color_en_esquina_superior_izquierda():
    from drowsiness_processor.alert.traffic_light import LightState as L, TrafficLightSnapshot as S
    from drowsiness_processor.visualization.traffic_light_visualizer import TrafficLightVisualizer, COLORS
    vis = TrafficLightVisualizer(red_border=False)   # el borde rojo pinta todo el marco a proposito
    for estado, elapsed in ((L.GREEN, 0.4), (L.YELLOW, 1.7), (L.RED, 3.1), (L.IDLE, 0.0)):
        frame = np.zeros((480, 640, 3), np.uint8)
        out = vis.draw(frame, S(state=estado, elapsed_s=elapsed, reason="x"))
        assert out is frame, "debe dibujar en sitio"
        r = max(24, int(480 * 0.075))
        cx, cy = 14 + r, 14 + r
        px = tuple(int(v) for v in frame[cy - r + 5, cx])    # punto dentro del circulo, lejos del texto
        assert px == COLORS[estado], f"{estado.value}: pixel {px} != {COLORS[estado]}"
        assert frame[440:, 400:].sum() == 0, "no debe pintar lejos de la esquina"
        assert frame[: 2 * r + 30, : 2 * r + 30].sum() > 0
    borde = np.zeros((480, 640, 3), np.uint8)         # en ROJO el marco completo se pinta de rojo
    TrafficLightVisualizer().draw(borde, S(state=L.RED, elapsed_s=3.0))
    assert tuple(int(v) for v in borde[240, 1]) == (0, 0, 255)


@test(S6)
def test_anclaje_arriba_derecha_y_entradas_invalidas():
    from drowsiness_processor.alert.traffic_light import LightState as L, TrafficLightSnapshot as S
    from drowsiness_processor.visualization.traffic_light_visualizer import TrafficLightVisualizer, COLORS
    vis = TrafficLightVisualizer(anchor="top_right")
    frame = np.zeros((480, 640, 3), np.uint8)
    vis.draw(frame, S(state=L.GREEN))
    r = max(24, int(480 * 0.075))
    assert tuple(int(v) for v in frame[14 + 5, 640 - 14 - r]) == COLORS[L.GREEN]
    for malo in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((50, 50), np.uint8)):
        vis.draw(malo, S(state=L.RED))               # no debe lanzar
    chico = np.zeros((60, 80, 3), np.uint8)
    vis.draw(chico, S(state=L.RED, possible_call=True, phone_visible=True))


# =============================================================================== 7. FACEMESH REAL
S7 = "7. Rostro real (MediaPipe)"
FOTOS = [("docs/face_mesh/full_points.png", (480, 480)),           # se redimensiona SIN deformar (misma proporcion)
         ("docs/face_mesh/face_mesh_frontal.jpg", (480, 720))]


def foto_frontal():
    for rel, size in FOTOS:
        ruta = os.path.join(ROOT, *rel.split("/"))
        if os.path.exists(ruta):
            img = cv2.imread(ruta)
            if img is not None:
                return cv2.resize(img, size)
    raise Skip("no hay una foto frontal en docs/face_mesh/ (full_points.png o face_mesh_frontal.jpg)")


def relajar_deteccion(face_mesh_processor):
    """SOLO para estas pruebas: umbral 0.3 para que la foto (trae la malla dibujada encima) se detecte siempre.
    Valida la integracion, no la calidad de deteccion de tu camara."""
    try:
        from drowsiness_processor.extract_points.face_mesh.face_mesh_processor import FaceMeshInference
        face_mesh_processor.inference = FaceMeshInference(0.3, 0.3)
    except Exception:
        pass


@test(S7)
def test_facemesh_real_entrega_478_landmarks_y_frente():
    from drowsiness_processor.extract_points.face_mesh.face_mesh_processor import FaceMeshProcessor
    from drowsiness_processor.head_pose.orientation import HeadOrientationClassifier, HeadOrientationConfig
    img = foto_frontal()
    ih, iw = img.shape[:2]
    fm = FaceMeshProcessor()
    relajar_deteccion(fm)
    _, ok, _ = fm.process(img)
    assert ok, "MediaPipe no detecto el rostro de la foto de prueba"
    lm = fm.last_landmarks
    assert lm is not None and lm.shape[0] >= 468 and lm.shape[1] == 3, getattr(lm, "shape", None)
    clf = HeadOrientationClassifier(HeadOrientationConfig(auto_calibrate=False))   # sin calibrar: angulos crudos
    r = clf.classify(lm, iw, ih)
    assert r.has_face and r.label == "frente", f"{r.label} yaw={r.yaw:.1f} pitch={r.pitch:.1f}"
    assert abs(r.yaw) < 20 and abs(r.pitch) < 20
    print(f"      (rostro real sin calibrar: yaw={r.yaw:.1f} pitch={r.pitch:.1f}; lo ideal es cerca de 0)")
    x1, y1, x2, y2 = r.face_bbox
    assert 0 <= x1 < x2 < iw and 0 <= y1 < y2 < ih


@test(S7)
def test_sin_rostro_last_landmarks_es_none():
    from drowsiness_processor.extract_points.face_mesh.face_mesh_processor import FaceMeshProcessor
    fm = FaceMeshProcessor()
    relajar_deteccion(fm)
    fm.process(foto_frontal())
    assert fm.last_landmarks is not None
    _, ok, _ = fm.process(np.zeros((480, 640, 3), np.uint8))
    assert not ok and fm.last_landmarks is None, "last_landmarks debe limpiarse cuando no hay rostro"


# =============================================================================== 8. SISTEMA COMPLETO
S8 = "8. Sistema completo (DrowsinessDetectionSystem)"
_SYSTEM = {}


def get_system(mute):
    if "s" not in _SYSTEM:
        from drowsiness_processor.main import DrowsinessDetectionSystem
        from drowsiness_processor.reports.main import DrowsinessReports
        from drowsiness_processor.alert.alarm import AlarmPlayer
        t0 = time.time()
        s = DrowsinessDetectionSystem()
        relajar_deteccion(s.points_extractor.face_mesh)
        # los reportes de la prueba van a un CSV temporal para no ensuciar tus reportes reales
        s.reports = DrowsinessReports(os.path.join(tempfile.mkdtemp(), "reporte_prueba.csv"))
        if mute:
            s.traffic_light.alarm = AlarmPlayer(cooldown_s=1.2, play_fn=lambda w, sr: None)
        _SYSTEM["s"] = s
        _SYSTEM["load_s"] = time.time() - t0
    return _SYSTEM["s"]


def b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


def reporte_a_dict(rep):
    """DrowsinessReports.generate_json_report() entrega un STRING JSON; antes del primer rostro es un dict."""
    if isinstance(rep, str):
        rep = json.loads(rep)
    assert isinstance(rep, dict), f"json_report con tipo inesperado: {type(rep)}"
    return rep


@test(S8)
def test_el_sistema_se_construye():
    get_system(ARGS.mute)
    assert hasattr(_SYSTEM["s"], "traffic_light") and hasattr(_SYSTEM["s"], "head_orientation"), \
        "main.py no tiene el semaforo integrado (ver instrucciones, paso 3)"


@test(S8)
def test_frames_con_rostro_calibra_y_queda_en_verde():
    s = get_system(ARGS.mute)
    img = foto_frontal()
    b = b64(img)
    t0 = time.time()
    for _ in range(22):
        orig, sketch, rep = s.run(b)
    fps = 22 / (time.time() - t0)
    assert isinstance(orig, np.ndarray) and orig.shape == img.shape and isinstance(sketch, np.ndarray)
    assert isinstance(rep, (str, dict)), f"json_report con tipo inesperado: {type(rep)}"
    json.dumps(rep)                                  # lo que enviara FastAPI por el WebSocket
    rep = reporte_a_dict(rep)
    assert "distraction" in rep, "json_report no trae la clave 'distraction'"
    assert "yawn" in rep and "pitch" in rep, "se perdio el reporte de somnolencia original al agregar el semaforo"
    d = rep["distraction"]
    assert d["state"] == "verde" and d["head_orientation"] == "frente", d
    r = max(24, int(min(img.shape[:2]) * 0.075))
    px = tuple(int(v) for v in orig[14 + 5, 14 + r])
    assert px == (0, 200, 0), f"no se dibujo el circulo verde en la esquina superior izquierda: {px}"
    print(f"      (rendimiento medido: {fps:.1f} FPS de pipeline completo en esta PC)")


@test(S8)
def test_celular_junto_a_la_cara_da_rojo_directo():
    s = get_system(ARGS.mute)
    img = foto_frontal()
    if not hasattr(s, "distraction_monitoring"):
        raise Skip("main.py no tiene distraction_monitoring()")
    s.points_extractor.process(img)                  # deja los landmarks de este fotograma listos
    lm = s.points_extractor.face_mesh.last_landmarks
    assert lm is not None, "no se detecto el rostro de la foto"
    ih, iw = img.shape[:2]
    x1, y1 = (lm[:, 0].min() * iw), (lm[:, 1].min() * ih)
    x2, y2 = (lm[:, 0].max() * iw), (lm[:, 1].max() * ih)
    snap = None
    for _ in range(3):
        s.points_extractor.process(img)
        snap = s.distraction_monitoring(img, [(x2 - 20, y1 + 80, x2 + 90, y1 + 260)])
    assert snap.possible_call and snap.state.value == "rojo", snap
    for _ in range(12):                              # el celular desaparece
        s.points_extractor.process(img)
        snap = s.distraction_monitoring(img, [])
        time.sleep(0.05)
    assert snap.state.value == "verde", f"no volvio a verde: {snap}"


@test(S8)
def test_rostro_perdido_da_rojo_y_pide_alarma():
    s = get_system(ARGS.mute)
    negro = np.zeros((H, W, 3), np.uint8)
    b = b64(negro)
    if not ARGS.mute:
        print("      (vas a escuchar 2 pitidos: es la alarma funcionando)")
    alarmas, ultimo, t0 = 0, None, time.time()
    while time.time() - t0 < 4.3:
        _, _, rep = s.run(b)
        ultimo = reporte_a_dict(rep)["distraction"]
        alarmas += ultimo["alarm_requested"]
        time.sleep(0.03)
    assert ultimo["state"] == "rojo" and ultimo["reason"] == "sin rostro", ultimo
    assert alarmas >= 2, f"solo {alarmas} pitidos pedidos en 4.3 s (rojo desde los 2.5 s)"


# =============================================================================== modos opcionales
def run_beep():
    from drowsiness_processor.alert.alarm import AlarmPlayer
    a = AlarmPlayer(cooldown_s=1.2)
    print("Sonido de alarma: 3 pitidos, uno cada ~1.2 s. Escuchalos ahora...")
    t0, n = time.time(), 0
    while n < 3 and time.time() - t0 < 6:
        if a.request():
            n += 1
            print(f"  pitido {n} pedido a los {time.time() - t0:.2f} s")
        time.sleep(0.02)                             # simula fotogramas: request() se llama muchas veces
    time.sleep(0.6)
    print(f"Backend de audio usado: {a.backend}")
    if a.backend != "sounddevice":
        print("AVISO: no se uso sounddevice; se uso el respaldo. Revisa PortAudio / dispositivo de salida.")
    return 0


def run_camera(index):
    from drowsiness_processor.main import DrowsinessDetectionSystem
    s = DrowsinessDetectionSystem()
    s.traffic_light_visualizer.show_debug = True
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara con indice {index}. Prueba con 0, 1 o 2.")
        return 1
    print("\nENSAYO EN VIVO  (q/ESC = salir, c = recalibrar)")
    print("1) Mira de frente ~2 s: el circulo gris muestra CALIBRANDO y luego pasa a verde.")
    print("2) Gira la cabeza a TU izquierda: en pantalla debe decir 'Cabeza: izquierda'. Luego a tu derecha y hacia abajo.")
    print("3) Manten la mirada desviada: 1 s -> amarillo, 2.5 s -> rojo + pitido cada ~1.2 s.")
    print("4) Acerca un celular a la oreja: rojo directo (si tu deteccion de celular esta conectada).")
    print("   En consola se imprime cada cambio de estado con los angulos yaw/pitch para ajustar umbrales.\n")
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("La camara dejo de entregar fotogramas")
            break
        frame, sketch, rep = s.frame_processing(frame)
        d = rep.get("distraction", {}) if isinstance(rep, dict) else {}
        key = (d.get("state"), d.get("head_orientation"), d.get("phone_visible"), d.get("possible_call"))
        if key != last:
            state_str = str(d.get("state") or "none")
            head_str = str(d.get("head_orientation") or "none")
            print(f"  estado={state_str:9s} cabeza={head_str:10s} "
                  f"celular={d.get('phone_visible')} llamada={d.get('possible_call')} "
                  f"yaw={d.get('yaw')} pitch={d.get('pitch')}")
            last = key
        cv2.imshow("Frame con semaforo (q=salir, c=recalibrar)", frame)
        cv2.imshow("Sketch", sketch)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("c"):
            s.recalibrate_head_pose()
            print("  -> recalibrando: mira al frente ~2 s")
    cap.release()
    cv2.destroyAllWindows()
    return 0
# =============================================================================== runner
def run_all():
    filas, fallos, avisos = [], 0, 0
    seccion = None
    for sec, nombre, fn in TESTS:
        if sec != seccion:
            print(f"\n== {sec}")
            seccion = sec
        t0 = time.time()
        try:
            fn()
            estado, detalle = "OK", ""
        except Skip as e:
            estado, detalle = "SALTADO", str(e)
        except Warn as e:
            estado, detalle, avisos = "AVISO", str(e), avisos + 1
        except AssertionError as e:
            estado, detalle, fallos = "FALLO", str(e) or "assert", fallos + 1
        except Exception:
            estado, detalle, fallos = "FALLO", traceback.format_exc(limit=4), fallos + 1
        print(f"  [{estado:7s}] {nombre}  ({time.time() - t0:.2f}s)")
        if detalle:
            print("      " + detalle.replace("\n", "\n      "))
        filas.append((sec, nombre, estado))
    ok = sum(1 for f in filas if f[2] == "OK")
    print("\n" + "=" * 70)
    print(f"RESUMEN: {ok} OK | {fallos} FALLO | {avisos} AVISO | {sum(1 for f in filas if f[2] == 'SALTADO')} SALTADO")
    if _SYSTEM.get("load_s") is not None:
        print(f"Carga del sistema completo: {_SYSTEM['load_s']:.1f} s")
    if fallos:
        print("HAY FALLOS: corrige antes de la demo (el mensaje de cada [FALLO] indica que revisar).")
        return 1
    if avisos:
        print("TODO FUNCIONA, con avisos: revisalos (normalmente es el audio).")
    else:
        print("TODO OK: listo para el ensayo en vivo (--camera) y la demo.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mute", action="store_true", help="no reproducir sonido real en las pruebas")
    ap.add_argument("--beep", action="store_true", help="solo probar el sonido de la alarma")
    ap.add_argument("--camera", nargs="?", const=0, type=int, metavar="INDICE",
                    help="ensayo en vivo con la webcam (por defecto indice 0)")
    ARGS = ap.parse_args()
    if ARGS.beep:
        sys.exit(run_beep())
    if ARGS.camera is not None:
        sys.exit(run_camera(ARGS.camera))
    sys.exit(run_all())
else:
    ARGS = argparse.Namespace(mute=True, beep=False, camera=None)
