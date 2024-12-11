import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QSlider, QSpinBox, QGridLayout, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import imutils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax.set_facecolor('#2b2b2b')
        self.figure.patch.set_facecolor('#2b2b2b')
        self.ax.grid(True, color='gray', alpha=0.3)
        self.ax.tick_params(colors='white')

    def update_histogram(self, image):
        self.ax.clear()
        if len(image.shape) == 3:
            colors = ('b', 'g', 'r')
            labels = ('Blue', 'Green', 'Red')
            for i, (color, label) in enumerate(zip(colors, labels)):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.ax.plot(hist, color=color, label=label, linewidth=2)
            self.ax.legend()
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            self.ax.plot(hist, color='white', linewidth=2)

        self.ax.set_xlim([0, 256])
        self.ax.set_ylim(bottom=0)  # Начинаем с нуля
        self.ax.grid(True, color='gray', alpha=0.3)
        self.ax.tick_params(colors='white')
        self.canvas.draw()


class ImageProcessor:
    @staticmethod
    def linear_contrast(image, alpha, beta):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def histogram_equalization_rgb(image):
        b, g, r = cv2.split(image)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge((b_eq, g_eq, r_eq))

    @staticmethod
    def histogram_equalization_hsv(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge((h, s, v_eq))
        return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    @staticmethod
    def detect_edges(image, threshold1, threshold2):
        return cv2.Canny(image, threshold1, threshold2)

    @staticmethod
    def detect_lines(image, rho, theta, threshold):
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        result = image.copy()

        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return result

    @staticmethod
    def detect_corners(image, max_corners, quality_level, min_distance):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance
        )
        result = image.copy()

        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), -1)

        return result


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Application")
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QComboBox {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
                min-width: 200px;
            }
            QSpinBox, QSlider {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555555;
            }
            QGroupBox {
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)

        self.original_image = None
        self.processed_image = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, stretch=3)

        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

        self.update_controls()
        self.resize(1600, 900)

    def create_left_panel(self):
        group_box = QGroupBox("Controls")
        layout = QVBoxLayout()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)

        layout.addWidget(QLabel("Processing Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Linear Contrast",
            "RGB Histogram Equalization",
            "HSV Histogram Equalization",
            "Edge Detection",
            "Line Detection",
            "Corner Detection"
        ])
        self.method_combo.currentIndexChanged.connect(self.update_controls)
        layout.addWidget(self.method_combo)

        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        layout.addWidget(self.params_widget)

        process_btn = QPushButton("Process Image")
        process_btn.clicked.connect(self.process_image)
        layout.addWidget(process_btn)

        layout.addStretch()
        group_box.setLayout(layout)
        return group_box

    def create_center_panel(self):
        group_box = QGroupBox("Images")
        layout = QGridLayout()

        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.addWidget(QLabel("Original Image:"))
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_layout.addWidget(self.original_label)
        layout.addWidget(original_container, 0, 0)

        processed_container = QWidget()
        processed_layout = QVBoxLayout(processed_container)
        processed_layout.addWidget(QLabel("Processed Image:"))
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        processed_layout.addWidget(self.processed_label)
        layout.addWidget(processed_container, 0, 1)

        group_box.setLayout(layout)
        return group_box

    def create_right_panel(self):
        group_box = QGroupBox("Histograms")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Original Histogram:"))
        self.original_histogram = HistogramWidget()
        layout.addWidget(self.original_histogram)

        layout.addWidget(QLabel("Processed Histogram:"))
        self.processed_histogram = HistogramWidget()
        layout.addWidget(self.processed_histogram)

        group_box.setLayout(layout)
        return group_box

    def update_controls(self):
        for i in reversed(range(self.params_layout.count())):
            self.params_layout.itemAt(i).widget().setParent(None)

        method = self.method_combo.currentText()

        if method == "Linear Contrast":
            alpha_container = QWidget()
            alpha_layout = QVBoxLayout(alpha_container)
            self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
            self.alpha_slider.setRange(10, 30)
            self.alpha_slider.setValue(10)
            alpha_layout.addWidget(QLabel("Contrast (alpha):"))
            alpha_layout.addWidget(self.alpha_slider)
            self.params_layout.addWidget(alpha_container)

            # Beta slider
            beta_container = QWidget()
            beta_layout = QVBoxLayout(beta_container)
            self.beta_slider = QSlider(Qt.Orientation.Horizontal)
            self.beta_slider.setRange(-50, 50)
            self.beta_slider.setValue(0)
            beta_layout.addWidget(QLabel("Brightness (beta):"))
            beta_layout.addWidget(self.beta_slider)
            self.params_layout.addWidget(beta_container)

        elif method == "Edge Detection":
            threshold1_container = QWidget()
            threshold1_layout = QVBoxLayout(threshold1_container)
            self.threshold1 = QSpinBox()
            self.threshold1.setRange(0, 255)
            self.threshold1.setValue(100)
            threshold1_layout.addWidget(QLabel("Threshold 1:"))
            threshold1_layout.addWidget(self.threshold1)
            self.params_layout.addWidget(threshold1_container)

            threshold2_container = QWidget()
            threshold2_layout = QVBoxLayout(threshold2_container)
            self.threshold2 = QSpinBox()
            self.threshold2.setRange(0, 255)
            self.threshold2.setValue(200)
            threshold2_layout.addWidget(QLabel("Threshold 2:"))
            threshold2_layout.addWidget(self.threshold2)
            self.params_layout.addWidget(threshold2_container)

        elif method == "Line Detection":
            params = [
                ("Rho:", "rho_spin", 1, 10, 1),
                ("Theta:", "theta_spin", 1, 180, 180),
                ("Threshold:", "threshold_spin", 0, 200, 100)
            ]

            for label_text, attr_name, min_val, max_val, default_val in params:
                container = QWidget()
                layout = QVBoxLayout(container)
                setattr(self, attr_name, QSpinBox())
                spin = getattr(self, attr_name)
                spin.setRange(min_val, max_val)
                spin.setValue(default_val)
                layout.addWidget(QLabel(label_text))
                layout.addWidget(spin)
                self.params_layout.addWidget(container)

        elif method == "Corner Detection":
            params = [
                ("Max Corners:", "max_corners", QSpinBox, 1, 100, 25),
                ("Quality Level:", "quality_level", QSlider, 1, 100, 10),
                ("Min Distance:", "min_distance", QSpinBox, 1, 100, 10)
            ]

            for label_text, attr_name, widget_type, min_val, max_val, default_val in params:
                container = QWidget()
                layout = QVBoxLayout(container)
                widget = widget_type()
                if widget_type == QSlider:
                    widget.setOrientation(Qt.Orientation.Horizontal)
                widget.setRange(min_val, max_val)
                widget.setValue(default_val)
                setattr(self, attr_name, widget)
                layout.addWidget(QLabel(label_text))
                layout.addWidget(widget)
                self.params_layout.addWidget(container)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if file_name:
            self.original_image = cv2.imread(file_name)
            self.display_image(self.original_image, self.original_label)
            self.original_histogram.update_histogram(self.original_image)
            self.processed_image = None
            self.processed_label.clear()
            self.processed_histogram.ax.clear()
            self.processed_histogram.canvas.draw()

    def process_image(self):
        if self.original_image is None:
            return

        method = self.method_combo.currentText()

        if method == "Linear Contrast":
            alpha = self.alpha_slider.value() / 10.0
            beta = self.beta_slider.value()
            self.processed_image = ImageProcessor.linear_contrast(
                self.original_image, alpha, beta)

        elif method == "RGB Histogram Equalization":
            self.processed_image = ImageProcessor.histogram_equalization_rgb(
                self.original_image)

        elif method == "HSV Histogram Equalization":
            self.processed_image = ImageProcessor.histogram_equalization_hsv(
                self.original_image)

        elif method == "Edge Detection":
            self.processed_image = ImageProcessor.detect_edges(
                self.original_image, self.threshold1.value(), self.threshold2.value())

        elif method == "Line Detection":
            self.processed_image = ImageProcessor.detect_lines(
                self.original_image,
                self.rho_spin.value(),
                np.pi / self.theta_spin.value(),
                self.threshold_spin.value()
            )

        elif method == "Corner Detection":
            self.processed_image = ImageProcessor.detect_corners(
                self.original_image,
                self.max_corners.value(),
                self.quality_level.value() / 100.0,
                self.min_distance.value()
            )

        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_label)
            self.processed_histogram.update_histogram(self.processed_image)

    def display_image(self, image, label):
        image = imutils.resize(image, width=500)
        height, width = image.shape[:2]

        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line,
                           QImage.Format.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line,
                           QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_label)
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_label)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    plt.style.use('dark_background')

    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec())