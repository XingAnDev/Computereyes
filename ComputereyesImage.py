import sys
from pathlib import Path

# ============================================================
# 先加载 PyTorch / YOLO
# 避免部分 Windows 环境下 Qt DLL 与 PyTorch DLL 加载顺序冲突
# ============================================================

from PIL import Image

from ultralytics import YOLO

import torch


# ============================================================
# 再加载 PyQt5
# ============================================================

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QStatusBar,
    QProgressBar,
)


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


# ============================================================
# 检查 CUDA
# ============================================================

print("=" * 60)
print("ComputerEyes")
print("=" * 60)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
print(f"GPU数   : {torch.cuda.device_count()}")

if torch.cuda.is_available():

    for i in range(
        torch.cuda.device_count()
    ):

        print(
            f"GPU {i}: "
            f"{torch.cuda.get_device_name(i)}"
        )

print(f"YOLO Device: {DEVICE}")

print("=" * 60)


# ============================================================
# YOLO 检测线程
# ============================================================

class DetectionThread(QThread):

    finished_signal = pyqtSignal(object, object)

    error_signal = pyqtSignal(str)

    status_signal = pyqtSignal(str)

    def __init__(
        self,
        model,
        image_path
    ):

        super().__init__()

        self.model = model

        self.image_path = image_path

    def run(self):

        try:

            self.status_signal.emit(
                "正在使用 RTX 5060 进行检测..."
            )

            results = self.model.predict(

                source=self.image_path,

                device=DEVICE,

                conf=CONFIDENCE,

                verbose=False,

            )

            result = results[0]

            # ====================================================
            # YOLO 自动画框
            #
            # 这里保持原样。
            # 检测框、颜色、标签全部由 YOLO result.plot()
            # 负责。
            # ====================================================

            annotated = result.plot()

            self.finished_signal.emit(
                result,
                annotated
            )

        except Exception as e:

            self.error_signal.emit(
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# 主窗口
# ============================================================

class ComputerEyes(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "ComputerEyes - YOLO11n"
        )

        self.resize(
            1400,
            900
        )

        self.setMinimumSize(
            1100,
            700
        )

        # ----------------------------------------------------
        # 数据
        # ----------------------------------------------------

        self.model = None

        self.current_image_path = None

        self.current_result = None

        self.original_pixmap = None

        self.result_pixmap = None

        self.detection_thread = None

        # ----------------------------------------------------
        # 创建界面
        # ----------------------------------------------------

        self.create_ui()

        # ----------------------------------------------------
        # 加载模型
        # ----------------------------------------------------

        self.load_model()


    # ========================================================
    # 创建 GUI
    # ========================================================

    def create_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        # ====================================================
        # ★ 黑色 / 白色主题
        # ====================================================

        self.setStyleSheet(

            """

            QMainWindow {
                background-color: #000000;
                color: #FFFFFF;
            }

            QWidget {
                background-color: #000000;
                color: #FFFFFF;
            }

            QLabel {
                color: #FFFFFF;
            }

            QGroupBox {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: bold;
            }

            QGroupBox::title {
                color: #FFFFFF;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #000000;
            }

            QPushButton {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 7px 14px;
            }

            QPushButton:hover {
                background-color: #1A1A1A;
                border: 1px solid #FFFFFF;
            }

            QPushButton:pressed {
                background-color: #2A2A2A;
            }

            QPushButton:disabled {
                background-color: #080808;
                color: #666666;
                border: 1px solid #222222;
            }

            QListWidget {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #333333;
                alternate-background-color: #080808;
                selection-background-color: #333333;
                selection-color: #FFFFFF;
            }

            QListWidget::item {
                color: #FFFFFF;
                padding: 5px;
            }

            QListWidget::item:selected {
                background-color: #333333;
                color: #FFFFFF;
            }

            QStatusBar {
                background-color: #000000;
                color: #FFFFFF;
                border-top: 1px solid #222222;
            }

            QProgressBar {
                background-color: #111111;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 4px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #FFFFFF;
            }

            """

        )

        # ====================================================
        # 主布局
        # ====================================================

        main_layout = QVBoxLayout(
            central_widget
        )

        main_layout.setContentsMargins(
            15,
            15,
            15,
            10
        )

        main_layout.setSpacing(
            10
        )


        # ====================================================
        # 标题
        # ====================================================

        header_layout = QHBoxLayout()

        title = QLabel(
            "ComputerEyes"
        )

        title_font = QFont()

        title_font.setPointSize(
            22
        )

        title_font.setBold(
            True
        )

        title.setFont(
            title_font
        )

        title.setStyleSheet(
            "color: #FFFFFF;"
        )


        subtitle = QLabel(
            "YOLO11n Object Detection"
        )

        subtitle.setStyleSheet(
            """
            color: #FFFFFF;
            font-size: 13px;
            """
        )


        header_layout.addWidget(
            title
        )

        header_layout.addWidget(
            subtitle
        )

        header_layout.addStretch()


        self.gpu_label = QLabel(
            "GPU: RTX 5060 Laptop GPU"
        )

        self.gpu_label.setStyleSheet(
            """
            color: #FFFFFF;
            font-size: 13px;
            font-weight: bold;
            """
        )

        header_layout.addWidget(
            self.gpu_label
        )

        main_layout.addLayout(
            header_layout
        )


        # ====================================================
        # 按钮
        # ====================================================

        button_layout = QHBoxLayout()


        self.upload_button = QPushButton(
            "📁  上传图片"
        )

        self.upload_button.setMinimumHeight(
            40
        )

        self.upload_button.setMinimumWidth(
            140
        )

        self.upload_button.clicked.connect(
            self.select_image
        )


        self.save_button = QPushButton(
            "💾  保存检测结果"
        )

        self.save_button.setMinimumHeight(
            40
        )

        self.save_button.setMinimumWidth(
            160
        )

        self.save_button.setEnabled(
            False
        )

        self.save_button.clicked.connect(
            self.save_result
        )


        button_layout.addWidget(
            self.upload_button
        )

        button_layout.addWidget(
            self.save_button
        )

        button_layout.addStretch()


        main_layout.addLayout(
            button_layout
        )


        # ====================================================
        # 图片区域
        # ====================================================

        image_layout = QHBoxLayout()

        image_layout.setSpacing(
            10
        )


        # ----------------------------------------------------
        # 原图
        # ----------------------------------------------------

        original_group = QGroupBox(
            "原始图片"
        )

        original_layout = QVBoxLayout(
            original_group
        )


        self.original_label = QLabel(
            "请选择一张图片"
        )

        self.original_label.setAlignment(
            Qt.AlignCenter
        )

        self.original_label.setMinimumSize(
            500,
            450
        )

        self.original_label.setStyleSheet(
            """

            QLabel {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #333333;
            }

            """
        )


        original_layout.addWidget(
            self.original_label
        )


        # ----------------------------------------------------
        # 检测结果
        # ----------------------------------------------------

        result_group = QGroupBox(
            "YOLO 检测结果"
        )

        result_layout = QVBoxLayout(
            result_group
        )


        self.result_label = QLabel(
            "检测结果会显示在这里"
        )

        self.result_label.setAlignment(
            Qt.AlignCenter
        )

        self.result_label.setMinimumSize(
            500,
            450
        )

        self.result_label.setStyleSheet(
            """

            QLabel {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #333333;
            }

            """
        )


        result_layout.addWidget(
            self.result_label
        )


        image_layout.addWidget(
            original_group,
            1
        )

        image_layout.addWidget(
            result_group,
            1
        )


        main_layout.addLayout(
            image_layout,
            1
        )


        # ====================================================
        # 检测信息
        # ====================================================

        info_layout = QHBoxLayout()


        # ----------------------------------------------------
        # 目标列表
        # ----------------------------------------------------

        detection_group = QGroupBox(
            "检测目标"
        )

        detection_layout = QVBoxLayout(
            detection_group
        )


        self.object_list = QListWidget()

        self.object_list.setMinimumHeight(
            150
        )

        detection_layout.addWidget(
            self.object_list
        )


        # ----------------------------------------------------
        # 模型信息
        # ----------------------------------------------------

        model_group = QGroupBox(
            "模型信息"
        )

        model_layout = QVBoxLayout(
            model_group
        )


        self.model_info = QLabel(

            "Model: YOLO11n\n"
            "Device: cuda:0\n"
            "GPU: RTX 5060 Laptop GPU\n"
            "Confidence: 0.25"

        )

        self.model_info.setStyleSheet(
            """

            color: #FFFFFF;
            font-family: Consolas;
            font-size: 12px;

            """
        )


        model_layout.addWidget(
            self.model_info
        )

        model_layout.addStretch()


        info_layout.addWidget(
            detection_group,
            2
        )

        info_layout.addWidget(
            model_group,
            1
        )


        main_layout.addLayout(
            info_layout
        )


        # ====================================================
        # 状态栏
        # ====================================================

        self.status_bar = QStatusBar()

        self.setStatusBar(
            self.status_bar
        )


        self.status_label = QLabel(
            "正在初始化..."
        )

        self.status_label.setStyleSheet(
            "color: #FFFFFF;"
        )


        self.status_bar.addWidget(
            self.status_label
        )


        self.progress = QProgressBar()

        self.progress.setMaximumWidth(
            180
        )

        self.progress.setRange(
            0,
            0
        )

        self.progress.hide()


        self.status_bar.addPermanentWidget(
            self.progress
        )


    # ========================================================
    # 加载 YOLO
    # ========================================================

    def load_model(self):

        model_file = Path(
            MODEL_PATH
        )


        if not model_file.exists():

            QMessageBox.critical(

                self,

                "模型不存在",

                f"找不到 YOLO 模型：\n\n"
                f"{MODEL_PATH}"

            )

            self.status_label.setText(
                "模型不存在"
            )

            return


        self.status_label.setText(
            "正在加载 YOLO11n..."
        )

        QApplication.processEvents()


        try:

            self.model = YOLO(
                MODEL_PATH
            )

            # ------------------------------------------------
            # 明确使用 RTX 5060
            # ------------------------------------------------

            self.model.to(
                DEVICE
            )


            self.status_label.setText(

                "YOLO11n 加载成功 | "
                "RTX 5060 | cuda:0"

            )


            print(
                "YOLO model loaded successfully."
            )


        except Exception as e:

            self.model = None


            QMessageBox.critical(

                self,

                "模型加载失败",

                f"YOLO 模型加载失败：\n\n{e}"

            )


            self.status_label.setText(
                "模型加载失败"
            )


    # ========================================================
    # 选择图片
    # ========================================================

    def select_image(self):

        if self.model is None:

            QMessageBox.warning(

                self,

                "模型未加载",

                "YOLO 模型还没有成功加载。"

            )

            return


        file_path, _ = QFileDialog.getOpenFileName(

            self,

            "选择图片",

            "",

            "Images "
            "(*.jpg *.jpeg *.png *.bmp *.webp)"

        )


        if not file_path:

            return


        self.current_image_path = file_path


        # ----------------------------------------------------
        # 读取原图
        # ----------------------------------------------------

        try:

            image = Image.open(
                file_path
            ).convert(
                "RGB"
            )


        except Exception as e:

            QMessageBox.critical(

                self,

                "图片打开失败",

                str(e)

            )

            return


        self.original_pixmap = (

            self.pil_to_pixmap(
                image
            )

        )


        self.update_image_display(

            self.original_label,

            self.original_pixmap

        )


        # ----------------------------------------------------
        # 清除旧结果
        # ----------------------------------------------------

        self.result_pixmap = None

        self.current_result = None

        self.result_label.clear()

        self.result_label.setText(
            "正在检测..."
        )


        self.object_list.clear()


        self.save_button.setEnabled(
            False
        )


        # ----------------------------------------------------
        # 禁用上传
        # ----------------------------------------------------

        self.upload_button.setEnabled(
            False
        )

        self.progress.show()


        self.status_label.setText(

            "正在使用 RTX 5060 "
            "进行 YOLO 检测..."

        )


        # ----------------------------------------------------
        # 创建检测线程
        # ----------------------------------------------------

        self.detection_thread = DetectionThread(

            self.model,

            file_path

        )


        self.detection_thread.finished_signal.connect(

            self.detection_finished

        )


        self.detection_thread.error_signal.connect(

            self.detection_error

        )


        self.detection_thread.status_signal.connect(

            self.update_status

        )


        self.detection_thread.finished.connect(

            self.detection_thread.deleteLater

        )


        self.detection_thread.start()


    # ========================================================
    # 检测完成
    # ========================================================

    def detection_finished(

        self,

        result,

        annotated

    ):

        self.current_result = result


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        annotated_rgb = annotated[
            :,
            :,
            ::-1
        ].copy()


        height, width, channels = (
            annotated_rgb.shape
        )


        bytes_per_line = (
            channels * width
        )


        image_bytes = annotated_rgb.tobytes()


        q_image = QImage(

            image_bytes,

            width,

            height,

            bytes_per_line,

            QImage.Format_RGB888

        )


        self.result_pixmap = (

            QPixmap.fromImage(
                q_image.copy()
            )

        )


        self.update_image_display(

            self.result_label,

            self.result_pixmap

        )


        # ----------------------------------------------------
        # 检测目标
        # ----------------------------------------------------

        self.object_list.clear()


        boxes = result.boxes


        if boxes is None or len(boxes) == 0:

            self.object_list.addItem(
                "没有检测到目标"
            )


        else:

            names = result.names


            for index, box in enumerate(

                boxes,

                start=1

            ):

                class_id = int(
                    box.cls[0]
                )


                confidence = float(
                    box.conf[0]
                )


                x1, y1, x2, y2 = (

                    box.xyxy[0].tolist()

                )


                class_name = names[
                    class_id
                ]


                text = (

                    f"{index:02d}  "
                    f"{class_name:<18} "
                    f"{confidence:.2%}    "
                    f"Box: "
                    f"({x1:.0f}, {y1:.0f}) "
                    f"→ "
                    f"({x2:.0f}, {y2:.0f})"

                )


                self.object_list.addItem(

                    QListWidgetItem(
                        text
                    )

                )


        # ----------------------------------------------------
        # 恢复 GUI
        # ----------------------------------------------------

        self.upload_button.setEnabled(
            True
        )

        self.save_button.setEnabled(
            True
        )

        self.progress.hide()


        count = (

            len(boxes)

            if boxes is not None

            else 0

        )


        self.status_label.setText(

            f"检测完成 | "
            f"目标数量: {count} | "
            f"RTX 5060 | "
            f"cuda:0"

        )


    # ========================================================
    # 检测错误
    # ========================================================

    def detection_error(

        self,

        error

    ):

        self.upload_button.setEnabled(
            True
        )

        self.progress.hide()


        self.status_label.setText(
            "检测失败"
        )


        QMessageBox.critical(

            self,

            "YOLO 检测失败",

            error

        )


    # ========================================================
    # 更新状态
    # ========================================================

    def update_status(

        self,

        text

    ):

        self.status_label.setText(
            text
        )


    # ========================================================
    # 保存结果
    # ========================================================

    def save_result(self):

        if self.current_result is None:

            QMessageBox.warning(

                self,

                "没有检测结果",

                "请先检测一张图片。"

            )

            return


        save_path, _ = (

            QFileDialog.getSaveFileName(

                self,

                "保存检测结果",

                "",

                "JPEG Image (*.jpg);;"
                "PNG Image (*.png)"

            )

        )


        if not save_path:

            return


        try:

            self.current_result.save(
                filename=save_path
            )


            QMessageBox.information(

                self,

                "保存成功",

                f"检测结果已保存：\n\n"
                f"{save_path}"

            )


        except Exception as e:

            QMessageBox.critical(

                self,

                "保存失败",

                str(e)

            )


    # ========================================================
    # PIL -> QPixmap
    # ========================================================

    @staticmethod
    def pil_to_pixmap(
        image
    ):

        image = image.convert(
            "RGB"
        )


        data = image.tobytes(
            "raw",
            "RGB"
        )


        q_image = QImage(

            data,

            image.width,

            image.height,

            image.width * 3,

            QImage.Format_RGB888

        )


        return QPixmap.fromImage(
            q_image.copy()
        )


    # ========================================================
    # 图片自适应显示
    # ========================================================

    @staticmethod
    def update_image_display(

        label,

        pixmap

    ):

        if pixmap is None:

            return


        scaled = pixmap.scaled(

            label.size(),

            Qt.KeepAspectRatio,

            Qt.SmoothTransformation

        )


        label.setPixmap(
            scaled
        )


    # ========================================================
    # 窗口缩放
    # ========================================================

    def resizeEvent(

        self,

        event

    ):

        super().resizeEvent(
            event
        )


        if self.original_pixmap:

            self.update_image_display(

                self.original_label,

                self.original_pixmap

            )


        if self.result_pixmap:

            self.update_image_display(

                self.result_label,

                self.result_pixmap

            )


# ============================================================
# 主函数
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setStyle(
        "Fusion"
    )

    window = ComputerEyes()

    window.show()

    sys.exit(
        app.exec_()
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()