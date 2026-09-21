import sys
import time
import ctypes

import numpy as np
import torch
import dxcam

from pynput import keyboard, mouse

from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
)

from PyQt5.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QFont,
)

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDoubleSpinBox,
    QSpinBox,
)

import win32gui
import win32con

from ultralytics import YOLO


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


# =========================================================
# 全局目标坐标
# =========================================================

GoalX = None
GoalY = None


# =========================================================
# Alt 目标定位
# =========================================================

ALT_KEYS = {
    keyboard.Key.alt,
    keyboard.Key.alt_l,
    keyboard.Key.alt_r,
    keyboard.Key.alt_gr,
}

mouse_ctrl = mouse.Controller()

_alt_held = False


def _on_press(key):
    global _alt_held

    if key in ALT_KEYS:

        # 防止 Alt 长按时不断重复移动
        if not _alt_held:

            _alt_held = True

            # 当前存在有效目标
            if GoalX is not None and GoalY is not None:

                mouse_ctrl.position = (
                    int(GoalX),
                    int(GoalY),
                )


def _on_release(key):
    global _alt_held

    if key in ALT_KEYS:
        _alt_held = False


# =========================================================
# 启动全局键盘监听
#
# 注意：
# 不要使用 _listener.join()
# 否则会阻塞 Qt 主程序
# =========================================================

_listener = keyboard.Listener(
    on_press=_on_press,
    on_release=_on_release,
)

_listener.daemon = True
_listener.start()


# =========================================================
# CUDA 性能设置
# =========================================================

if torch.cuda.is_available():

    # 对固定尺寸的卷积推理通常有帮助
    torch.backends.cudnn.benchmark = True

    # 对部分 CUDA 运算允许使用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass


# =========================================================
# COCO 80 类
# =========================================================

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# =========================================================
# 之前版本的框颜色
# =========================================================

CLASS_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 180, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 80, 80),
    (80, 180, 255),
    (180, 80, 255),
]


# =========================================================
# 获取所有窗口
# =========================================================

def get_all_windows():

    windows = []
    seen = set()

    shell_classes = {
        "Progman",
        "WorkerW",
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
    }

    def add_window(hwnd):

        if hwnd in seen:
            return

        seen.add(hwnd)

        try:

            if not win32gui.IsWindow(hwnd):
                return

            if not win32gui.IsWindowVisible(hwnd):
                return

            left, top, right, bottom = (
                win32gui.GetWindowRect(hwnd)
            )

            width = right - left
            height = bottom - top

            if width < 100 or height < 100:
                return

            class_name = win32gui.GetClassName(
                hwnd
            )

            if class_name in shell_classes:
                return

            title = win32gui.GetWindowText(
                hwnd
            ).strip()

            if not title:
                title = f"[{class_name}]"

            windows.append(
                {
                    "hwnd": hwnd,
                    "title": title,
                    "class": class_name,
                }
            )

        except Exception:
            pass

    def enum_child_proc(hwnd, _):
        add_window(hwnd)
        return True

    def enum_proc(hwnd, _):

        add_window(hwnd)

        try:

            win32gui.EnumChildWindows(
                hwnd,
                enum_child_proc,
                None,
            )

        except Exception:
            pass

        return True

    try:

        win32gui.EnumWindows(
            enum_proc,
            None,
        )

    except Exception:
        pass

    windows.sort(
        key=lambda x: (
            x["title"].lower(),
            x["hwnd"],
        )
    )

    return windows


# =========================================================
# 获取窗口客户区
# =========================================================

def get_window_region(hwnd):

    try:

        left, top, right, bottom = (
            win32gui.GetClientRect(hwnd)
        )

        if right <= left or bottom <= top:
            return None

        point = win32gui.ClientToScreen(
            hwnd,
            (0, 0),
        )

        screen_x = point[0]
        screen_y = point[1]

        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        return (
            screen_x,
            screen_y,
            screen_x + width,
            screen_y + height,
        )

    except Exception:
        return None


# =========================================================
# YOLO 检测线程
# =========================================================

class DetectionThread(QThread):

    detections_ready = pyqtSignal(
        list,
        float,
        float,
    )

    capture_info = pyqtSignal(
        float
    )

    error = pyqtSignal(
        str
    )

    def __init__(
        self,
        camera,
        source_type,
        hwnd,
        confidence,
        selected_classes,
        image_size,
        parent=None,
    ):

        super().__init__(parent)

        self.camera = camera
        self.source_type = source_type
        self.hwnd = hwnd
        self.confidence = confidence
        self.selected_classes = selected_classes

        # 用户从 GUI 设置的 Image Size
        self.image_size = image_size

        self.running = True
        self.model = None

        # =====================================================
        # 随机目标跟踪
        # =====================================================

        self.current_target = None

        # GoalX / GoalY 已经改成全局变量
        # 不再使用 self.GoalX / self.GoalY

        # -----------------------------------------------------
        # FPS 统计
        # -----------------------------------------------------

        self.yolo_counter = 0
        self.yolo_timer = time.perf_counter()

        self.capture_counter = 0
        self.capture_timer = time.perf_counter()

        self.yolo_fps = 0.0
        self.capture_fps = 0.0

        # -----------------------------------------------------
        # Overlay 限频
        # -----------------------------------------------------

        self.last_overlay_emit = 0.0

        self.overlay_interval = (
            1.0 / OVERLAY_FPS
        )

        # -----------------------------------------------------
        # 窗口区域
        # -----------------------------------------------------

        self.region = None

    def stop(self):

        self.running = False

    # =====================================================
    # 加载模型
    # =====================================================

    def load_model(self):

        print("=" * 50)
        print("Loading YOLO")
        print("=" * 50)

        self.model = YOLO(
            MODEL_PATH
        )

        print(
            "YOLO Device:",
            DEVICE,
        )

        print(
            "Image Size:",
            self.image_size,
        )

        print(
            "Warming up YOLO..."
        )

        dummy = np.zeros(
            (
                self.image_size,
                self.image_size,
                3,
            ),
            dtype=np.uint8,
        )

        with torch.inference_mode():

            self.model.predict(
                source=dummy,
                device=DEVICE,
                imgsz=self.image_size,
                conf=self.confidence,
                classes=self.selected_classes,
                quantize=16,
                verbose=False,
            )

        # -----------------------------------------------------
        # CUDA 同步一次
        # 确保 Warmup 真正结束
        # -----------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(
            "Warmup complete."
        )

        print(
            "Model loaded successfully."
        )

    # =====================================================
    # Run
    # =====================================================

    def run(self):

        global GoalX, GoalY

        try:

            self.load_model()

            # =================================================
            # DXCam
            # =================================================

            if self.source_type == "screen":

                self.region = None

                self.camera.start(
                    target_fps=TARGET_FPS,
                    video_mode=False,
                )

            else:

                self.region = get_window_region(
                    self.hwnd
                )

                if self.region is None:

                    self.error.emit(
                        "无法获取窗口客户区"
                    )

                    return

                self.camera.start(
                    region=self.region,
                    target_fps=TARGET_FPS,
                    video_mode=False,
                )

            # =================================================
            # 实时检测循环
            # =================================================

            while self.running:

                # -------------------------------------------------
                # 永远只拿最新帧
                # -------------------------------------------------

                frame = (
                    self.camera.get_latest_frame()
                )

                if frame is None:

                    time.sleep(
                        0.0005
                    )

                    continue

                # -------------------------------------------------
                # Capture FPS
                # -------------------------------------------------

                self.capture_counter += 1

                now = time.perf_counter()

                capture_elapsed = (
                    now - self.capture_timer
                )

                if capture_elapsed >= 1.0:

                    self.capture_fps = (
                        self.capture_counter
                        / capture_elapsed
                    )

                    self.capture_counter = 0
                    self.capture_timer = now

                    self.capture_info.emit(
                        self.capture_fps
                    )

                # -------------------------------------------------
                # YOLO 推理
                # -------------------------------------------------

                inference_start = (
                    time.perf_counter()
                )

                with torch.inference_mode():

                    results = self.model.predict(
                        source=frame,
                        device=DEVICE,
                        conf=self.confidence,
                        classes=self.selected_classes,
                        imgsz=self.image_size,
                        quantize=16,
                        verbose=False,
                    )

                # CUDA 异步执行时，读取结果前自然会同步
                # 所以这里的时间更接近真实推理+结果等待时间

                inference_ms = (
                    time.perf_counter()
                    - inference_start
                ) * 1000.0

                detections = []

                if results:

                    result = results[0]
                    boxes = result.boxes

                    if boxes is not None:

                        # =================================================
                        # 关键优化：
                        #
                        # boxes.data 一次性 GPU → CPU
                        # =================================================

                        data = (
                            boxes.data
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        if data.size:

                            for row in data:

                                x1 = float(
                                    row[0]
                                )

                                y1 = float(
                                    row[1]
                                )

                                x2 = float(
                                    row[2]
                                )

                                y2 = float(
                                    row[3]
                                )

                                confidence = float(
                                    row[4]
                                )

                                class_id = int(
                                    row[5]
                                )

                                detections.append(
                                    {
                                        "x1": x1,
                                        "y1": y1,
                                        "x2": x2,
                                        "y2": y2,
                                        "class_id": class_id,
                                        "confidence": confidence,
                                    }
                                )

                # =================================================
                # 如果检测的是窗口：
                #
                # YOLO 坐标是窗口局部坐标
                # Overlay 是整个屏幕
                #
                # 所以需要加窗口屏幕偏移
                # =================================================

                if (
                    self.source_type == "window"
                    and self.region is not None
                ):

                    offset_x = self.region[0]
                    offset_y = self.region[1]

                    for detection in detections:

                        detection["x1"] += offset_x
                        detection["x2"] += offset_x

                        detection["y1"] += offset_y
                        detection["y2"] += offset_y

                # =================================================
                # 随机选择目标
                # =================================================

                if detections:

                    # 第一次没有目标时
                    # 随机选择一个

                    if self.current_target is None:

                        self.current_target = detections[
                            np.random.randint(
                                0,
                                len(detections)
                            )
                        ]

                    else:

                        # 当前目标中心点

                        old_x = (
                            self.current_target["x1"]
                            + self.current_target["x2"]
                        ) / 2.0

                        old_y = (
                            self.current_target["y1"]
                            + self.current_target["y2"]
                        ) / 2.0

                        # 找距离当前目标最近的检测目标

                        nearest_target = None
                        nearest_distance = float(
                            "inf"
                        )

                        for detection in detections:

                            new_x = (
                                detection["x1"]
                                + detection["x2"]
                            ) / 2.0

                            new_y = (
                                detection["y1"]
                                + detection["y2"]
                            ) / 2.0

                            distance = (
                                (new_x - old_x) ** 2
                                + (new_y - old_y) ** 2
                            )

                            if distance < nearest_distance:

                                nearest_distance = distance
                                nearest_target = detection

                        # -------------------------------------------------
                        # 判断原目标是否还存在
                        # -------------------------------------------------

                        if nearest_target is not None:

                            MAX_TARGET_DISTANCE = 150 ** 2

                            if (
                                nearest_distance
                                <= MAX_TARGET_DISTANCE
                            ):

                                # 原目标还在

                                self.current_target = (
                                    nearest_target
                                )

                            else:

                                # 原目标消失
                                # 从当前所有目标中重新随机选择

                                self.current_target = detections[
                                    np.random.randint(
                                        0,
                                        len(detections)
                                    )
                                ]

                else:

                    # 当前完全没有目标

                    self.current_target = None

                    # 清空全局目标坐标
                    GoalX = None
                    GoalY = None

                # =================================================
                # 计算目标中心点
                # =================================================

                if self.current_target is not None:

                    GoalX = (
                        self.current_target["x1"]
                        + self.current_target["x2"]
                    ) / 2.0

                    GoalY = (
                        self.current_target["y1"]
                        + self.current_target["y2"]
                    ) / 2.0

                # =================================================
                # YOLO FPS
                # =================================================

                self.yolo_counter += 1

                now = time.perf_counter()

                yolo_elapsed = (
                    now - self.yolo_timer
                )

                if yolo_elapsed >= 1.0:

                    self.yolo_fps = (
                        self.yolo_counter
                        / yolo_elapsed
                    )

                    self.yolo_counter = 0
                    self.yolo_timer = now

                # =================================================
                # Overlay 限频
                #
                # YOLO 可以尽可能快地跑
                # 但 GUI 不需要每一帧都 repaint
                # =================================================

                now = time.perf_counter()

                if (
                    now
                    - self.last_overlay_emit
                    >= self.overlay_interval
                ):

                    self.last_overlay_emit = now

                    self.detections_ready.emit(
                        detections,
                        self.yolo_fps,
                        inference_ms,
                    )

        except Exception as e:

            self.error.emit(
                f"{type(e).__name__}: {e}"
            )

        finally:

            # 检测线程退出时清空目标坐标
            GoalX = None
            GoalY = None

            try:
                self.camera.stop()
            except Exception:
                pass


# =========================================================
# Detection Overlay
# =========================================================

class DetectionOverlay(QWidget):

    def __init__(
        self,
        screen_geometry,
        parent=None,
    ):

        super().__init__(
            parent
        )

        self.detections = []

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground,
            True,
        )

        self.setAttribute(
            Qt.WA_TransparentForMouseEvents,
            True,
        )

        self.setAttribute(
            Qt.WA_NoSystemBackground,
            True,
        )

        self.setGeometry(
            screen_geometry
        )

        self.enable_click_through()

        # -----------------------------------------------------
        # 尝试让 Overlay 不进入屏幕捕获
        # -----------------------------------------------------

        try:

            hwnd = int(
                self.winId()
            )

            WDA_EXCLUDEFROMCAPTURE = 0x11

            ctypes.windll.user32.SetWindowDisplayAffinity(
                hwnd,
                WDA_EXCLUDEFROMCAPTURE,
            )

        except Exception:
            pass

    # =====================================================
    # Windows 点击穿透
    # =====================================================

    def enable_click_through(self):

        try:

            hwnd = int(
                self.winId()
            )

            style = (
                ctypes.windll.user32.GetWindowLongW(
                    hwnd,
                    win32con.GWL_EXSTYLE,
                )
            )

            style |= (
                win32con.WS_EX_LAYERED
                | win32con.WS_EX_TRANSPARENT
                | win32con.WS_EX_TOOLWINDOW
                | win32con.WS_EX_NOACTIVATE
            )

            ctypes.windll.user32.SetWindowLongW(
                hwnd,
                win32con.GWL_EXSTYLE,
                style,
            )

        except Exception:
            pass

    # =====================================================
    # 更新检测结果
    # =====================================================

    def set_detections(
        self,
        detections,
    ):

        self.detections = detections

        self.update()

    def clear(self):

        self.detections = []

        self.update()

    # =====================================================
    # 绘制
    # =====================================================

    def paintEvent(
        self,
        event,
    ):

        if not self.detections:
            return

        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.Antialiasing,
            False,
        )

        # -----------------------------------------------------
        # 保持之前版本字体
        # -----------------------------------------------------

        font = QFont(
            "Segoe UI",
            9,
        )

        painter.setFont(
            font
        )

        metrics = (
            painter.fontMetrics()
        )

        for detection in self.detections:

            x1 = int(
                detection["x1"]
            )

            y1 = int(
                detection["y1"]
            )

            x2 = int(
                detection["x2"]
            )

            y2 = int(
                detection["y2"]
            )

            class_id = int(
                detection["class_id"]
            )

            confidence = float(
                detection["confidence"]
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            # =================================================
            # 类别名称
            # =================================================

            if (
                0 <= class_id
                < len(COCO_CLASSES)
            ):

                name = COCO_CLASSES[
                    class_id
                ]

            else:

                name = str(
                    class_id
                )

            # =================================================
            # 之前版本颜色
            # =================================================

            color_tuple = CLASS_COLORS[
                class_id
                % len(CLASS_COLORS)
            ]

            color = QColor(
                color_tuple[0],
                color_tuple[1],
                color_tuple[2],
            )

            # =================================================
            # 检测框
            # =================================================

            painter.setBrush(
                Qt.NoBrush
            )

            painter.setPen(
                QPen(
                    color,
                    2,
                )
            )

            painter.drawRect(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            )

            # =================================================
            # 标签
            # =================================================

            label = (
                f"{name} "
                f"{confidence * 100:.1f}%"
            )

            text_width = (
                metrics.horizontalAdvance(
                    label
                )
                + 8
            )

            text_height = (
                metrics.height()
                + 4
            )

            label_x = x1

            label_y = (
                y1
                - text_height
            )

            if label_y < 0:
                label_y = y1

            # -------------------------------------------------
            # 黑色背景
            # -------------------------------------------------

            painter.setPen(
                Qt.NoPen
            )

            painter.setBrush(
                QBrush(
                    QColor(
                        0,
                        0,
                        0,
                        220,
                    )
                )
            )

            painter.drawRect(
                label_x,
                label_y,
                text_width,
                text_height,
            )

            # -------------------------------------------------
            # 白色文字
            # -------------------------------------------------

            painter.setPen(
                QColor(
                    255,
                    255,
                    255,
                )
            )

            painter.drawText(
                label_x + 4,
                label_y
                + metrics.ascent()
                + 2,
                label,
            )

        painter.end()


# =========================================================
# Main Window
# =========================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "ComputerEyes"
        )

        self.resize(
            1050,
            760,
        )

        self.setStyleSheet(
            """
            QWidget {
                background: #000000;
                color: #ffffff;
                font-family: "Segoe UI";
            }

            QGroupBox {
                border: 1px solid #444444;
                margin-top: 10px;
                padding: 12px;
                color: #ffffff;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #ffffff;
            }

            QPushButton {
                background: #000000;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 7px 14px;
            }

            QPushButton:hover {
                background: #222222;
            }

            QPushButton:pressed {
                background: #333333;
            }

            QComboBox {
                background: #000000;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 5px;
            }

            QComboBox QAbstractItemView {
                background: #000000;
                color: #ffffff;
                selection-background-color: #333333;
            }

            QListWidget {
                background: #000000;
                color: #ffffff;
                border: 1px solid #444444;
            }

            QDoubleSpinBox {
                background: #000000;
                color: #ffffff;
                border: 1px solid #666666;
            }

            QSpinBox {
                background: #000000;
                color: #ffffff;
                border: 1px solid #666666;
            }

            QCheckBox {
                color: #ffffff;
            }

            QLabel {
                color: #ffffff;
            }
            """
        )

        # =====================================================
        # DXCam
        # =====================================================

        self.camera = dxcam.create(
            output_idx=0,
            output_color="RGB",
        )

        self.thread = None
        self.overlay = None
        self.windows = []

        # =====================================================
        # UI
        # =====================================================

        self.build_ui()

        self.create_overlay()

        self.refresh_windows()

    # =====================================================
    # Overlay
    # =====================================================

    def create_overlay(self):

        screen = (
            QApplication.primaryScreen()
        )

        if screen is None:
            return

        geometry = (
            screen.geometry()
        )

        self.overlay = DetectionOverlay(
            geometry
        )

        self.overlay.hide()

    # =====================================================
    # UI
    # =====================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        # -----------------------------------------------------
        # 标题
        # -----------------------------------------------------

        title = QLabel(
            "ComputerEyes"
        )

        title.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 28px;
                font-weight: bold;
            }
            """
        )

        main_layout.addWidget(
            title
        )

        subtitle = QLabel(
            "YOLO11n Real-Time Object Detection"
        )

        subtitle.setStyleSheet(
            """
            QLabel {
                color: #aaaaaa;
                font-size: 13px;
            }
            """
        )

        main_layout.addWidget(
            subtitle
        )

        author = QLabel(
            "XingAnDev"
        )

        author.setStyleSheet(
            """
            QLabel {
                color: #777777;
                font-size: 12px;
            }
            """
        )

        main_layout.addWidget(
            author
        )

        # =====================================================
        # System
        # =====================================================

        system_box = QGroupBox(
            "System"
        )

        system_layout = QGridLayout(
            system_box
        )

        if torch.cuda.is_available():

            gpu_name = (
                torch.cuda.get_device_name(
                    0
                )
            )

            cuda_version = (
                torch.version.cuda
            )

        else:

            gpu_name = "CPU"
            cuda_version = "N/A"

        system_layout.addWidget(
            QLabel("GPU:"),
            0,
            0,
        )

        system_layout.addWidget(
            QLabel(gpu_name),
            0,
            1,
        )

        system_layout.addWidget(
            QLabel("PyTorch:"),
            1,
            0,
        )

        system_layout.addWidget(
            QLabel(torch.__version__),
            1,
            1,
        )

        system_layout.addWidget(
            QLabel("CUDA:"),
            2,
            0,
        )

        system_layout.addWidget(
            QLabel(cuda_version),
            2,
            1,
        )

        main_layout.addWidget(
            system_box
        )

        # =====================================================
        # Capture Source
        # =====================================================

        capture_box = QGroupBox(
            "Capture Source"
        )

        capture_layout = QVBoxLayout(
            capture_box
        )

        self.source_combo = QComboBox()

        self.source_combo.addItem(
            "Entire Screen",
            None,
        )

        capture_layout.addWidget(
            self.source_combo
        )

        refresh_button = QPushButton(
            "Refresh Windows"
        )

        refresh_button.clicked.connect(
            self.refresh_windows
        )

        capture_layout.addWidget(
            refresh_button
        )

        main_layout.addWidget(
            capture_box
        )

        # =====================================================
        # Detection
        # =====================================================

        detection_box = QGroupBox(
            "Detection"
        )

        detection_layout = QHBoxLayout(
            detection_box
        )

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        detection_layout.addWidget(
            QLabel("Confidence:")
        )

        self.confidence_spin = (
            QDoubleSpinBox()
        )

        self.confidence_spin.setRange(
            0.01,
            1.00,
        )

        self.confidence_spin.setSingleStep(
            0.01
        )

        self.confidence_spin.setValue(
            DEFAULT_CONFIDENCE
        )

        detection_layout.addWidget(
            self.confidence_spin
        )

        # -----------------------------------------------------
        # Image Size
        # -----------------------------------------------------

        detection_layout.addWidget(
            QLabel("Image Size:")
        )

        self.image_size_spin = QSpinBox()

        self.image_size_spin.setRange(
            320,
            2048,
        )

        self.image_size_spin.setSingleStep(
            32
        )

        self.image_size_spin.setValue(
            DEFAULT_IMAGE_SIZE
        )

        detection_layout.addWidget(
            self.image_size_spin
        )

        main_layout.addWidget(
            detection_box
        )

        # =====================================================
        # Classes
        # =====================================================

        class_box = QGroupBox(
            "Object Classes"
        )

        class_layout = QVBoxLayout(
            class_box
        )

        class_buttons = QHBoxLayout()

        select_all_button = QPushButton(
            "Select All"
        )

        clear_all_button = QPushButton(
            "Clear All"
        )

        select_all_button.clicked.connect(
            self.select_all_classes
        )

        clear_all_button.clicked.connect(
            self.clear_all_classes
        )

        class_buttons.addWidget(
            select_all_button
        )

        class_buttons.addWidget(
            clear_all_button
        )

        class_layout.addLayout(
            class_buttons
        )

        self.class_list = QListWidget()

        for name in COCO_CLASSES:

            item = QListWidgetItem(
                name
            )

            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
            )

            if name == "person":

                item.setCheckState(
                    Qt.Checked
                )

            else:

                item.setCheckState(
                    Qt.Unchecked
                )

            self.class_list.addItem(
                item
            )

        class_layout.addWidget(
            self.class_list
        )

        main_layout.addWidget(
            class_box
        )

        # =====================================================
        # Performance
        # =====================================================

        stats_box = QGroupBox(
            "Performance"
        )

        stats_layout = QGridLayout(
            stats_box
        )

        self.capture_fps_label = QLabel(
            "Capture FPS: 0.0"
        )

        self.yolo_fps_label = QLabel(
            "YOLO FPS: 0.0"
        )

        self.inference_label = QLabel(
            "Inference: 0.00 ms"
        )

        self.targets_label = QLabel(
            "Targets: 0"
        )

        stats_layout.addWidget(
            self.capture_fps_label,
            0,
            0,
        )

        stats_layout.addWidget(
            self.yolo_fps_label,
            0,
            1,
        )

        stats_layout.addWidget(
            self.inference_label,
            1,
            0,
        )

        stats_layout.addWidget(
            self.targets_label,
            1,
            1,
        )

        main_layout.addWidget(
            stats_box
        )

        # =====================================================
        # Controls
        # =====================================================

        control_layout = QHBoxLayout()

        self.start_button = QPushButton(
            "Start"
        )

        self.stop_button = QPushButton(
            "Stop"
        )

        self.start_button.clicked.connect(
            self.start_detection
        )

        self.stop_button.clicked.connect(
            self.stop_detection
        )

        self.stop_button.setEnabled(
            False
        )

        control_layout.addWidget(
            self.start_button
        )

        control_layout.addWidget(
            self.stop_button
        )

        main_layout.addLayout(
            control_layout
        )

        # =====================================================
        # Status
        # =====================================================

        self.status_label = QLabel(
            "Ready"
        )

        main_layout.addWidget(
            self.status_label
        )

    # =====================================================
    # Refresh Windows
    # =====================================================

    def refresh_windows(self):

        self.windows = (
            get_all_windows()
        )

        current_hwnd = (
            self.source_combo.currentData()
        )

        self.source_combo.clear()

        self.source_combo.addItem(
            "Entire Screen",
            None,
        )

        for window in self.windows:

            hwnd = window["hwnd"]
            title = window["title"]

            self.source_combo.addItem(
                f"{title} (HWND: {hwnd})",
                hwnd,
            )

        if current_hwnd is not None:

            index = (
                self.source_combo.findData(
                    current_hwnd
                )
            )

            if index >= 0:

                self.source_combo.setCurrentIndex(
                    index
                )

        self.status_label.setText(
            f"Found {len(self.windows)} windows"
        )

    # =====================================================
    # Class Selection
    # =====================================================

    def select_all_classes(self):

        for i in range(
            self.class_list.count()
        ):

            self.class_list.item(
                i
            ).setCheckState(
                Qt.Checked
            )

    def clear_all_classes(self):

        for i in range(
            self.class_list.count()
        ):

            self.class_list.item(
                i
            ).setCheckState(
                Qt.Unchecked
            )

    def get_selected_classes(self):

        selected = []

        for i in range(
            self.class_list.count()
        ):

            item = (
                self.class_list.item(i)
            )

            if (
                item.checkState()
                == Qt.Checked
            ):

                selected.append(i)

        return selected

    # =====================================================
    # Start
    # =====================================================

    def start_detection(self):

        global GoalX, GoalY

        if self.thread is not None:
            return

        selected_classes = (
            self.get_selected_classes()
        )

        if not selected_classes:

            self.status_label.setText(
                "No classes selected"
            )

            return

        hwnd = (
            self.source_combo.currentData()
        )

        if hwnd is None:

            source_type = "screen"

        else:

            source_type = "window"

        confidence = (
            self.confidence_spin.value()
        )

        # -----------------------------------------------------
        # 获取 GUI 中的 Image Size
        # -----------------------------------------------------

        image_size = (
            self.image_size_spin.value()
        )

        # -----------------------------------------------------
        # 清空旧目标坐标
        # -----------------------------------------------------

        GoalX = None
        GoalY = None

        # -----------------------------------------------------
        # 清空旧框
        # -----------------------------------------------------

        if self.overlay is not None:

            self.overlay.clear()

            self.overlay.show()

            self.overlay.raise_()

        # -----------------------------------------------------
        # 创建线程
        # -----------------------------------------------------

        self.thread = DetectionThread(
            camera=self.camera,
            source_type=source_type,
            hwnd=hwnd,
            confidence=confidence,
            selected_classes=selected_classes,
            image_size=image_size,
        )

        self.thread.detections_ready.connect(
            self.on_detections
        )

        self.thread.capture_info.connect(
            self.on_capture_fps
        )

        self.thread.error.connect(
            self.on_error
        )

        self.thread.finished.connect(
            self.on_thread_finished
        )

        self.thread.start()

        self.start_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            True
        )

        self.status_label.setText(
            f"Running... Image Size: {image_size}"
        )

    # =====================================================
    # Stop
    # =====================================================

    def stop_detection(self):

        if self.thread is None:
            return

        self.thread.stop()

        self.status_label.setText(
            "Stopping..."
        )

    # =====================================================
    # Detection Result
    # =====================================================

    def on_detections(
        self,
        detections,
        yolo_fps,
        inference_ms,
    ):

        # -----------------------------------------------------
        # 更新 Overlay
        # -----------------------------------------------------

        if self.overlay is not None:

            self.overlay.set_detections(
                detections
            )

        # -----------------------------------------------------
        # 更新统计
        # -----------------------------------------------------

        self.yolo_fps_label.setText(
            f"YOLO FPS: {yolo_fps:.1f}"
        )

        self.inference_label.setText(
            f"Inference: {inference_ms:.2f} ms"
        )

        self.targets_label.setText(
            f"Targets: {len(detections)}"
        )

    # =====================================================
    # Capture FPS
    # =====================================================

    def on_capture_fps(
        self,
        fps,
    ):

        self.capture_fps_label.setText(
            f"Capture FPS: {fps:.1f}"
        )

    # =====================================================
    # Error
    # =====================================================

    def on_error(
        self,
        message,
    ):

        self.status_label.setText(
            message
        )

    # =====================================================
    # Thread Finished
    # =====================================================

    def on_thread_finished(self):

        global GoalX, GoalY

        GoalX = None
        GoalY = None

        self.thread = None

        if self.overlay is not None:

            self.overlay.clear()
            self.overlay.hide()

        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Stopped"
        )

        self.capture_fps_label.setText(
            "Capture FPS: 0.0"
        )

        self.yolo_fps_label.setText(
            "YOLO FPS: 0.0"
        )

        self.inference_label.setText(
            "Inference: 0.00 ms"
        )

        self.targets_label.setText(
            "Targets: 0"
        )

    # =====================================================
    # Close
    # =====================================================

    def closeEvent(
        self,
        event,
    ):

        global GoalX, GoalY

        GoalX = None
        GoalY = None

        if self.thread is not None:

            self.thread.stop()
            self.thread.wait()

            self.thread = None

        if self.overlay is not None:

            self.overlay.clear()
            self.overlay.hide()
            self.overlay.close()

            self.overlay = None

        try:

            self.camera.stop()

        except Exception:
            pass

        # -----------------------------------------------------
        # 停止键盘监听
        # -----------------------------------------------------

        try:

            _listener.stop()

        except Exception:
            pass

        event.accept()


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 50)
    print("ComputerEyes")
    print("=" * 50)

    print(
        "GPU Count:",
        torch.cuda.device_count()
    )

    for i in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {i}:",
            torch.cuda.get_device_name(i)
        )

    print()

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec_()
    )


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    main()