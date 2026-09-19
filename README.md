# Detección de Somnolencia y Distracción al Conducir

Sistema de visión por computadora para la mitigación de accidentes por distracción al conducir mediante IA en tiempo real.

Proyecto final de Inteligencia Artificial - Universidad Nacional de Ingeniería (UNI), Nicaragua.

---

## Integrantes

- Áreas Pérez David Jeshua
- Mairena Sierras Kevin Adonis
- Urbina Trigueros Allan Said

**Grupo:** 4T1-SIS-S
**Docente:** Ing. Cristóbal Jaime Garay
**Fecha:** Managua, 2026

---

## Descripción

Sistema de visión por computadora que analiza en tiempo real imágenes provenientes de una cámara orientada hacia el conductor. Detecta:

- Somnolencia: parpadeo prolongado, micro-sueño, bostezos
- Distracción visual: orientación de cabeza (frente / izquierda / derecha / abajo)
- Uso del celular: detección del teléfono y estimación de "posible llamada"
- Semáforo de atención: verde (atento), amarillo (distraído 1s), rojo (distraído 2.5s + alarma sonora)

Basado en la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining).

---

## Características

- Procesamiento en tiempo real con MediaPipe FaceMesh (478 landmarks faciales)
- Detección de celular con YOLOv8n preentrenado sobre COCO (clase 67 = cell phone)
- Estimación de orientación de cabeza con cv2.solvePnP (tolerante a lentes)
- Calibración automática: los primeros 15 fotogramas con rostro frontal definen el cero del conductor
- Semáforo de 3 colores con umbrales configurables (verde / amarillo / rojo)
- Alarma sonora no bloqueante con cooldown de 1.2 s
- Interfaz gráfica con Flet
- Backend modular con FastAPI + WebSocket
- Reportes en CSV y JSON

---

## Arquitectura

```
+-----------------+   WebSocket    +----------------------+
|   Flet (GUI)    | <------------> |   FastAPI backend    |
|   main.py       |  ws://.../ws   |   app.py             |
+-----------------+                +----------------------+
                                            |
                                            v
                             +----------------------------+
                             |  DrowsinessDetectionSystem |
                             |  MediaPipe + YOLOv8n       |
                             +----------------------------+
```

---

## Requisitos

- Python 3.10 o superior (recomendado 3.11)
- Cámara web funcional
- Windows / Linux / macOS
- Aproximadamente 3 GB de espacio en disco (PyTorch + modelos)
- Conexión a internet (solo para la primera descarga de modelos)

Librerías principales:
- mediapipe - landmarks faciales
- ultralytics - YOLOv8n para detección de celular
- opencv-python - procesamiento de imagen
- flet - interfaz gráfica
- fastapi + uvicorn - backend
- sounddevice - alarma sonora

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AllanUrbina/driver_fatigue_detection.git
cd driver_fatigue_detection
```

### 2. Crear y activar entorno virtual

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
pip install ultralytics
```

Si matplotlib==3.9.1 falla al instalar, usa:
```bash
pip install matplotlib==3.9.2
pip install -r requirements.txt
```

### 4. Ajustes necesarios (si aplica)

Si tu webcam está en otro índice distinto a 0:

```bash
# En gui/pages/drowsiness_page.py cambia:
# cv2.VideoCapture(1)  ->  cv2.VideoCapture(0)
```

Si tienes problemas de conexión WebSocket en Windows:

```bash
# En gui/pages/drowsiness_page.py cambia:
# ws://localhost:8000/ws  ->  ws://127.0.0.1:8000/ws
```

---

## Ejecución

Necesitas DOS terminales separadas.

### Terminal 1 - Backend (servidor)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Debe mostrar:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 - Interfaz gráfica (Flet)

```bash
flet main.py
```

### Uso

1. Se abre la ventana "Educare IA"
2. Clic en Welcome
3. Presiona Tab + Enter hasta llegar a la pantalla de detección
4. Clic en Start - se activa la cámara
5. Mira al frente aproximadamente 2 segundos para que el sistema calibre la posición de tu cabeza
6. Prueba:
   - Mueve la cabeza a un lado - amarillo (1 s) - rojo (2.5 s) + alarma
   - Levanta un celular frente a la cámara - aparece un rectángulo amarillo/rojo
   - Acércalo a tu oreja - "possible call" + alarma inmediata

---

## Dockerización (opcional)

```bash
docker build -t drowsiness-server .
docker run -d -p 8000:8000 --name drowsiness-server drowsiness-server
```

Importante: la alarma sonora no funciona dentro de Docker (el contenedor no tiene salida de audio). Para la demo ejecuta uvicorn directamente en la PC.

Detener el contenedor:
```bash
docker stop drowsiness-server
```

---

## Reportes

El sistema genera automáticamente:

- drowsiness_processor/reports/august/drowsiness_report.csv - registro acumulativo
- JSON con el estado actual enviado por WebSocket en cada fotograma

Campos registrados: timestamp, parpadeo, micro-sueño, bostezo, pitch, celular, semáforo.

---

## Verificación previa a la demo

```bash
python tests/test_distraction_system.py --mute       # pruebas sin sonido
python tests/test_distraction_system.py --beep       # probar solo la alarma
python tests/test_distraction_system.py --camera 0   # ensayo en vivo
```

Código de salida 0 = todo OK. 1 = hay fallos, no hagas la demo hasta corregir.

---

## Solución de problemas

| Error | Causa | Solución |
|---|---|---|
| ModuleNotFoundError: cv2 | Venv no activado | Activa el entorno virtual |
| Camera index out of range | Índice de cámara mal | Cambia VideoCapture(1) a VideoCapture(0) |
| TimeoutError: handshake | WebSocket mal apuntado | Cambia localhost a 127.0.0.1 |
| [Errno 10048] | Puerto 8000 ocupado | Get-Process python \| Stop-Process -Force |
| ImportError: PhoneDetector | Archivos de phone_detection vacíos | Rellena detector.py, processing.py, visualization.py |
| matplotlib error | Falta compilador C++ | pip install matplotlib==3.9.2 |

---

## Metodología CRISP-DM

El proyecto sigue las 6 fases de CRISP-DM:

1. Entendimiento del negocio - problema de seguridad vial
2. Entendimiento de los datos - imágenes y video de cabina
3. Preparación de los datos - extracción de fotogramas, etiquetado
4. Modelado - MediaPipe + YOLOv8n
5. Evaluación - métricas y pruebas bajo condiciones reales
6. Despliegue - prototipo funcional con interfaz gráfica

---

## Licencia

Proyecto académico - Universidad Nacional de Ingeniería (UNI), Nicaragua, 2026.

---

## Créditos

Basado en el proyecto original de Aprende e Ingenia (https://github.com/AprendeIngenia/driver_fatigue_detection), con adaptaciones para detección de celular (YOLOv8n), orientación de cabeza (solvePnP), semáforo de atención y alarma sonora.

---

## Contacto

- GitHub: https://github.com/AllanUrbina
- Repositorio: https://github.com/AllanUrbina/driver_fatigue_detection
