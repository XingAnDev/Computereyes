# Configuration

Before running the project, **please check the configuration section in the Python file you are going to use**.

The project currently has two main programs:

* `ComputereyesImage.py` — image-based YOLO detection
* `ComputereyesLIVE.py` — real-time detection using DXCam and a live overlay

The configuration sections are intentionally located near the beginning of each file so that they are easy to find and modify.

---

# `ComputereyesImage.py` Configuration

The configuration section is located at approximately **lines 41–57**.

```python
# ============================================================
# 配置
# ============================================================

MODEL_PATH = r"D:\Desktop\Workspace\YOLO\yolo11n.pt"

# PyTorch 实际枚举：
#
# cuda:0 = NVIDIA GeForce RTX 5060 Laptop GPU
#
# 注意：
# 不能根据 Windows 任务管理器里的 GPU0/GPU1
# 来直接决定 CUDA 编号。
DEVICE = "cuda:0"

CONFIDENCE = 0.25
```

## `MODEL_PATH`

```python
MODEL_PATH = r"D:\Desktop\Workspace\YOLO\yolo11n.pt"
```

This specifies the location of the YOLO model file.

The example path above is the path used on the development computer.

**You must change this path if `yolo11n.pt` is located somewhere else on your computer.**

For example:

```python
MODEL_PATH = r"C:\Users\YourName\Documents\ComputerEyes\yolo11n.pt"
```

or:

```python
MODEL_PATH = r"D:\AI\Models\yolo11n.pt"
```

The path must point to an existing YOLO model file.

If the path is incorrect, the program will not be able to load the model.

### Important

The `r` before the string is intentional:

```python
r"D:\AI\Models\yolo11n.pt"
```

It creates a Python raw string and prevents Windows backslashes from being interpreted as escape sequences.

---

## `DEVICE`

```python
DEVICE = "cuda:0"
```

This specifies the device used for YOLO inference.

The development environment uses:

```text
cuda:0
```

which corresponds to:

```text
NVIDIA GeForce RTX 5060 Laptop GPU
```

### Important: CUDA numbering is not Windows Task Manager numbering

Do **not** assume that:

```text
Windows Task Manager GPU 0
```

must correspond to:

```text
CUDA cuda:0
```

These are two different device numbering systems.

PyTorch enumerates CUDA-compatible NVIDIA GPUs independently.

You can check the actual CUDA devices detected by PyTorch with:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

For example:

```text
CUDA available: True
GPU count: 1
0 NVIDIA GeForce RTX 5060 Laptop GPU
```

In this situation:

```python
DEVICE = "cuda:0"
```

is correct.

If your computer has multiple NVIDIA GPUs, the CUDA device number may be different.

---

## `CONFIDENCE`

```python
CONFIDENCE = 0.25
```

This is the minimum confidence threshold used for object detection.

The value:

```text
0.25
```

means that detections with a confidence score below approximately 25% are filtered out.

For example:

```python
CONFIDENCE = 0.25
```

allows more potential detections.

A higher value such as:

```python
CONFIDENCE = 0.50
```

is more restrictive.

A lower value such as:

```python
CONFIDENCE = 0.15
```

is less restrictive.

Changing this value creates a trade-off between detecting more potential objects and filtering out weaker detections.

---

# `ComputereyesLIVE.py` Configuration

The configuration section is located at approximately **lines 48–66**.

```python
# =========================================================
# 配置
# =========================================================

MODEL_PATH = r"D:\Desktop\Workspace\YOLO\yolo11n.pt"

DEVICE = "cuda:0"

# GUI 默认 Image Size
DEFAULT_IMAGE_SIZE = 1024

# DXCam 后台采集目标
TARGET_FPS = 165

DEFAULT_CONFIDENCE = 0.25

# Overlay 最大刷新频率
# 不限制 YOLO，只限制 GUI 绘制
OVERLAY_FPS = 90
```

There are more settings in the live version because real-time detection has to control both the detection pipeline and the display/overlay pipeline.

---

## `MODEL_PATH`

```python
MODEL_PATH = r"D:\Desktop\Workspace\YOLO\yolo11n.pt"
```

This specifies the YOLO model used by the live detector.

Just like `ComputereyesImage.py`, this path is specific to the development environment.

**Change it if the model is stored somewhere else.**

For example:

```python
MODEL_PATH = r"D:\AI\Models\yolo11n.pt"
```

Make sure that the specified file exists before starting the program.

---

## `DEVICE`

```python
DEVICE = "cuda:0"
```

This specifies the CUDA device used by YOLO.

The development computer uses:

```text
cuda:0
```

which PyTorch reports as:

```text
NVIDIA GeForce RTX 5060 Laptop GPU
```

Again, this number is determined by PyTorch's CUDA device enumeration.

It should **not** be selected by looking at the GPU numbering shown by Windows Task Manager.

You can check your system with:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

---

## `DEFAULT_IMAGE_SIZE`

```python
DEFAULT_IMAGE_SIZE = 1024
```

This specifies the default image size used by the live YOLO detection configuration.

The current default value is:

```text
1024
```

This value affects the resolution at which the image/frame is processed by the detection pipeline.

A larger image size can preserve more visual detail, which may help with smaller objects.

However, increasing the processing resolution also increases the amount of computation required.

For example:

```python
DEFAULT_IMAGE_SIZE = 640
```

uses a smaller inference size.

While:

```python
DEFAULT_IMAGE_SIZE = 1280
```

uses a larger inference size.

The best value depends on:

* GPU performance
* Source resolution
* Target object size
* Required detection accuracy
* Desired detection speed

The default development configuration is:

```python
DEFAULT_IMAGE_SIZE = 1024
```

---

## `TARGET_FPS`

```python
TARGET_FPS = 165
```

This controls the target frame rate for the **DXCam background capture**.

The current configuration is:

```text
165 FPS
```

The purpose of this setting is to control how frequently the live input is captured.

It is separate from the overlay rendering limit.

For example:

```python
TARGET_FPS = 60
```

targets 60 FPS capture.

```python
TARGET_FPS = 120
```

targets 120 FPS capture.

```python
TARGET_FPS = 165
```

targets 165 FPS capture.

A higher capture rate can provide more frequent frames, but it also increases the amount of data that must be captured and processed.

The practical maximum depends on:

* Monitor refresh rate
* Capture source
* DXCam performance
* CPU performance
* GPU performance
* Image resolution
* YOLO inference time

---

## `DEFAULT_CONFIDENCE`

```python
DEFAULT_CONFIDENCE = 0.25
```

This is the default YOLO confidence threshold used by the live detection system.

The default value is:

```text
0.25
```

A detection must meet the configured confidence threshold before it is accepted.

Lower values may allow more detections through.

Higher values are more restrictive.

For example:

```python
DEFAULT_CONFIDENCE = 0.50
```

requires a higher confidence score.

The appropriate value depends on the target environment and detection requirements.

---

## `OVERLAY_FPS`

```python
OVERLAY_FPS = 90
```

This controls the maximum refresh rate of the GUI/overlay rendering.

The important distinction is:

> **`OVERLAY_FPS` limits the overlay drawing frequency. It does not limit YOLO inference.**

The current development configuration is:

```text
OVERLAY_FPS = 90
```

This means the overlay is limited to a maximum of approximately 90 refreshes per second.

This is intentionally separate from:

```python
TARGET_FPS = 165
```

The two settings control different parts of the pipeline.

### Capture

```text
TARGET_FPS = 165
```

controls the target capture rate.

### Overlay

```text
OVERLAY_FPS = 90
```

limits how frequently the graphical overlay is refreshed.

Therefore, the current configuration can be summarized as:

```text
DXCam capture target:
165 FPS

Overlay rendering limit:
90 FPS

YOLO:
Not limited by OVERLAY_FPS
```

This separation allows the capture/detection pipeline and the graphical display pipeline to operate independently.

---

# Configuration Summary

## `ComputereyesImage.py`

| Setting      |           Default | Purpose                        |
| ------------ | ----------------: | ------------------------------ |
| `MODEL_PATH` | `yolo11n.pt` path | Location of YOLO model         |
| `DEVICE`     |          `cuda:0` | CUDA device used for inference |
| `CONFIDENCE` |            `0.25` | Detection confidence threshold |

---

## `ComputereyesLIVE.py`

| Setting              |           Default | Purpose                           |
| -------------------- | ----------------: | --------------------------------- |
| `MODEL_PATH`         | `yolo11n.pt` path | Location of YOLO model            |
| `DEVICE`             |          `cuda:0` | CUDA device used for inference    |
| `DEFAULT_IMAGE_SIZE` |            `1024` | Default YOLO image/inference size |
| `TARGET_FPS`         |             `165` | DXCam background capture target   |
| `DEFAULT_CONFIDENCE` |            `0.25` | Default YOLO confidence threshold |
| `OVERLAY_FPS`        |              `90` | Maximum overlay refresh rate      |

---

# Recommended Configuration Procedure

Before running either program:

### Step 1 — Locate the model

Make sure you have:

```text
yolo11n.pt
```

available on your computer.

### Step 2 — Open the Python file

For image detection:

```text
ComputereyesImage.py
```

For live detection:

```text
ComputereyesLIVE.py
```

### Step 3 — Find the configuration section

For `ComputereyesImage.py`:

```text
Lines 41–57
```

For `ComputereyesLIVE.py`:

```text
Lines 48–66
```

### Step 4 — Change `MODEL_PATH`

Set it to the actual location of your model.

### Step 5 — Check `DEVICE`

Verify the CUDA device using PyTorch.

### Step 6 — Adjust detection settings

If necessary, modify:

```text
CONFIDENCE
DEFAULT_CONFIDENCE
DEFAULT_IMAGE_SIZE
TARGET_FPS
OVERLAY_FPS
```

according to your hardware and requirements.

### Step 7 — Start the program

Only start the program after verifying the configuration.

---

# Overall Project Workflow

The project is divided into two different processing modes.

## Image Detection

`ComputereyesImage.py` processes a static image.

The workflow is:

```text
User
 │
 │ Select / provide image
 ▼
ComputereyesImage.py
 │
 ▼
Load configuration
 │
 ├── MODEL_PATH
 ├── DEVICE
 └── CONFIDENCE
 │
 ▼
Load YOLO11n
 │
 ▼
Initialize PyTorch CUDA
 │
 ▼
Read image
 │
 ▼
Preprocess image
 │
 ▼
YOLO11n inference
 │
 ▼
Generate detections
 │
 ▼
Apply confidence filtering
 │
 ▼
Draw / process detection results
 │
 ▼
Display result
```

This mode is useful for testing the YOLO model independently from the real-time capture system.

---

# Live Detection

`ComputereyesLIVE.py` adds continuous frame capture and overlay rendering.

The workflow is:

```text
                    ┌─────────────────────┐
                    │  Configuration       │
                    │                     │
                    │ MODEL_PATH          │
                    │ DEVICE              │
                    │ IMAGE_SIZE          │
                    │ TARGET_FPS          │
                    │ CONFIDENCE          │
                    │ OVERLAY_FPS         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Load YOLO11n        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Initialize CUDA     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DXCam Capture       │
                    │ Target: 165 FPS     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Capture Frame       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ YOLO11n Inference   │
                    │ CUDA GPU            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Detection Results   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Overlay Rendering   │
                    │ Maximum: 90 FPS     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Next Frame          │
                    └──────────┬──────────┘
                               │
                               └──────────────► Repeat
```

---

# Capture, Detection and Overlay Are Separate

One important design principle of the live version is that the following operations are not treated as the same thing:

```text
Capture
Detection
Overlay
```

The current configuration is:

```text
Capture target:       165 FPS
YOLO inference:       GPU accelerated
Overlay maximum:       90 FPS
```

`OVERLAY_FPS` therefore does **not** mean that YOLO only runs at 90 FPS.

It only limits how frequently the graphical overlay is refreshed.

This separation is useful because the capture, inference, and rendering stages have different performance characteristics.

---

# Processing Pipeline

The simplified real-time pipeline is:

```text
DXCam
  │
  ▼
Frame Capture
  │
  ▼
Image Preparation
  │
  ▼
YOLO11n
  │
  ▼
CUDA / NVIDIA GPU
  │
  ▼
Detection Results
  │
  ▼
Bounding Boxes / Detection Data
  │
  ▼
Overlay
  │
  ▼
Screen
```

The loop then continues with the next captured frame.

---

# Performance Considerations

The three main performance-related settings are:

```python
DEFAULT_IMAGE_SIZE = 1024
TARGET_FPS = 165
OVERLAY_FPS = 90
```

These settings affect different stages of the application.

### Higher `DEFAULT_IMAGE_SIZE`

Advantages:

* More image detail available to YOLO
* Potentially better detection of small objects

Disadvantages:

* Higher GPU workload
* Higher memory usage
* Potentially lower inference speed

### Higher `TARGET_FPS`

Advantages:

* More frequent frame capture
* Lower time between captured frames

Disadvantages:

* Higher capture workload
* More frames entering the processing pipeline
* Higher CPU/GPU workload depending on the rest of the pipeline

### Higher `OVERLAY_FPS`

Advantages:

* More frequent visual updates

Disadvantages:

* Higher rendering workload

Because these values control different stages, they can be tuned independently.

---

# First-Time Setup Checklist

Before running the project for the first time:

```text
[ ] Python installed
[ ] PyTorch installed
[ ] TorchVision installed
[ ] Ultralytics installed
[ ] PyQt5 installed if required by the selected program
[ ] Pillow installed
[ ] NVIDIA driver installed
[ ] CUDA-enabled PyTorch verified
[ ] yolo11n.pt downloaded
[ ] MODEL_PATH configured
[ ] DEVICE verified
[ ] Confidence threshold checked
[ ] Live capture settings checked
[ ] Overlay settings checked
```

Then verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

A successful NVIDIA configuration should report:

```text
CUDA: True
```

followed by the detected NVIDIA GPU name.

---

# Important

The paths and values shown in the configuration section are examples based on the development environment.

**Do not assume that the original paths will exist on your computer.**

In particular, always check:

```python
MODEL_PATH
```

before running the program.

For CUDA:

```python
DEVICE = "cuda:0"
```

should only be used when PyTorch reports that the corresponding CUDA device exists.

For the live version, also review:

```python
DEFAULT_IMAGE_SIZE
TARGET_FPS
DEFAULT_CONFIDENCE
OVERLAY_FPS
```

before starting real-time detection.
You are free to use, modify, and redistribute this software
for non-commercial purposes.

Commercial use, sale, or distribution for profit is prohibited.

This software is provided "AS IS", without warranty of any kind.
The author is not responsible for any damage, loss, or problems
resulting from the use of this software.
