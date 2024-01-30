import sys
from datetime import datetime

import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import os
from comfyui_api import transform_image
from PIL import Image
import numpy as np
import configparser

setting = configparser.ConfigParser()
setting.read('config.ini')
workflow_dir = setting['DIR']['WORKFLOW_DIR']
result_dir = setting['DIR']['RESULT_DIR']


os.makedirs(os.path.join(result_dir,'ori'), exist_ok=True)
os.makedirs(os.path.join(result_dir,'full'), exist_ok=True)
os.makedirs(os.path.join(result_dir,'processed'), exist_ok=True)

modes = dict(setting['MODE_LIST'])


class ImageProcessor(QThread):
    finished = pyqtSignal(object)  # Signal to indicate processing is done

    def __init__(self, img_path, option):
        super().__init__()
        self.img_path = img_path
        self.option = option


    def run(self):
        file_name = self.img_path.split('\\')[-1]

        if modes[self.option] in os.listdir(workflow_dir):
            workflow_path = os.path.join(workflow_dir, modes[self.option])
            img = transform_image(workflow_path, self.img_path)

        else:
            img = Image.open(self.img_path)

        img.save(os.path.join(result_dir, 'processed', file_name))
        def get_concat_v(im1, im2):
            n_width = im1.width
            n_height = n_width * im2.height // im2.width
            im2 = im2.resize((n_width, n_height), Image.LANCZOS)
            dst = Image.new('RGB', (im1.width, im1.height+im2.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (0, im1.height))
            # dst.paste(im3, (im1.width*2, 0))
            return dst

        ori = Image.open(self.img_path)
        full_img = get_concat_v(ori, img)
        full_img.save(os.path.join(result_dir,'full', file_name))


        self.finished.emit(full_img)


class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Timer for updating the webcam feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # Timer for processing animation
        self.processing_timer = QTimer(self)
        self.processing_timer.timeout.connect(self.update_processing_animation)
        self.processing_animation_stage = 0
        self.processing_animation_running = False

    def initUI(self):
        self.setWindowTitle("Enhanced Photo Booth")

        self.setStyleSheet("""
                    QWidget {
                        background-color: #f2f2f2;
                        color: #333;
                        font-size: 14px;  # Larger font size
                    }
                    QPushButton {
                        background-color: #4CAF50;
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        color: white;
                        padding: 15px 32px;
                        font-size: 16px;
                        margin: 4px 2px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                    QLabel {
                        font-size: 16px;
                        border: 1px solid #ddd;  # Add a border to labels if needed
                    }
                """)

        # Styling
        # self.setStyleSheet("background-color: #f2f2f2; color: #333;")
        button_style = "QPushButton { background-color: #4CAF50; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;} QPushButton:hover { background-color: #45a049;}"
        label_style = "QLabel { font-size: 16px; }"

        # Webcam feed label
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")

        # Dropdown for resize options
        self.resize_dropdown = QComboBox(self)
        self.resize_dropdown.addItems(list(modes.keys()))
        self.resize_dropdown.setStyleSheet("QComboBox { combobox-popup: 0; }")

        # Button for taking a photo
        self.btn_capture = QPushButton('Take a photo', self)
        self.btn_capture.setStyleSheet(button_style)
        self.btn_capture.clicked.connect(self.capture_image)

        # Label for showing the processed image
        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("background-color: black;")

        # Label for "Processing..."
        self.processing_label = QLabel('')
        self.processing_label.setStyleSheet(label_style)

        # Set layout
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(self.btn_capture)
        controls_layout.addWidget(self.resize_dropdown)

        main_layout.addWidget(self.image_label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.processed_image_label)
        main_layout.addWidget(self.processing_label)

        self.setLayout(main_layout)

    @pyqtSlot()
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to format suitable for PyQt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0],
                           frame.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))

    def capture_image(self):
        # Get the current frame from webcam
        ret, frame = self.cap.read()
        if ret:
            # Start processing animation
            self.processing_animation_stage = 0
            self.processing_animation_running = True
            self.processing_timer.start(500)

            # Process the image
            resize_option = self.resize_dropdown.currentText()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            file_date = datetime.today().strftime("%Y_%m%d_%H_%M_%S")
            img_path = os.path.join(result_dir, 'ori', "face_" + file_date + ".png")
            frame.save(img_path)

            self.image_processor = ImageProcessor(img_path, resize_option)
            self.image_processor.finished.connect(self.on_image_processed)
            self.image_processor.start()

    @pyqtSlot()
    def update_processing_animation(self):
        if self.processing_animation_running:
            self.processing_label.setText("Processing" + "." * self.processing_animation_stage)
            self.processing_animation_stage = (self.processing_animation_stage + 1) % 4
        else:
            self.processing_label.setText("")
            self.processing_timer.stop()

    def on_image_processed(self, img):
        # Stop processing animation
        self.processing_animation_running = False

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Convert to format suitable for PyQt and display
        resized_image = QImage(img.data, img.shape[1], img.shape[0],
                               img.strides[0], QImage.Format_RGB888).rgbSwapped()
        self.processed_image_label.setPixmap(QPixmap.fromImage(resized_image))
        self.processing_timer.stop()

    def closeEvent(self, event):
        self.cap.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WebcamApp()
    ex.show()
    sys.exit(app.exec_())
