import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox,
                             QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os


class OrthophotoGenerator:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    def load_images(self, folder_path):
        img_paths = sorted(glob(os.path.join(folder_path, "*.jpg"))) + \
                    sorted(glob(os.path.join(folder_path, "*.jpeg"))) + \
                    sorted(glob(os.path.join(folder_path, "*.png")))

        images = []
        for path in tqdm(img_paths, desc="Загрузка изображений"):
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                if w > 2000 or h > 2000:
                    img = cv2.resize(img, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
                images.append(img)
        return images

    def create_orthophoto(self, images):
        if len(images) < 2:
            raise ValueError("Нужно минимум 2 изображения")

        status, result = self.stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return result
        else:
            return self.manual_stitching(images)

    def manual_stitching(self, images):
        base_img = images[0]
        for i in tqdm(range(1, len(images)), desc="Ручное сшивание"):
            gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)

            if des1 is not None and des2 is not None:
                matches = self.matcher.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]

                if len(good) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        h, w = base_img.shape[:2]
                        warped = cv2.warpPerspective(images[i], H, (w * 2, h * 2))
                        mask = np.zeros((h * 2, w * 2), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, np.int32(dst_pts * 2), 255)
                        base_img = cv2.seamlessClone(warped, cv2.resize(base_img, (w * 2, h * 2)), mask, (w, h),
                                                     cv2.NORMAL_CLONE)
        return base_img


class OrthoProcessor(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, input_folder, output_path):
        super().__init__()
        self.input_folder = input_folder
        self.output_path = output_path
        self.cancel_process = False

    def run(self):
        try:
            generator = OrthophotoGenerator()
            self.progress_signal.emit(10, "Загрузка изображений...")
            images = generator.load_images(self.input_folder)

            if len(images) < 2:
                self.error_signal.emit("Необходимо минимум 2 изображения")
                return

            self.progress_signal.emit(40, "Сшивание изображений...")
            result = generator.create_orthophoto(images)

            if self.cancel_process:
                return

            self.progress_signal.emit(90, "Сохранение результата...")
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            cv2.imwrite(self.output_path, result)

            self.finished_signal.emit(result)
            self.progress_signal.emit(100, "Готово!")

        except Exception as e:
            self.error_signal.emit(f"Ошибка: {str(e)}")


class OrthoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор ортофотопланов")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        input_group = QGroupBox("Настройки ввода")
        input_layout = QVBoxLayout()

        self.input_label = QLabel("Папка с изображениями: не выбрана")
        btn_input = QPushButton("Выбрать папку")
        btn_input.clicked.connect(self.select_input_folder)

        self.output_label = QLabel("Файл результата: не выбран")
        btn_output = QPushButton("Выбрать файл результата")
        btn_output.clicked.connect(self.select_output_file)

        input_layout.addWidget(self.input_label)
        input_layout.addWidget(btn_input)
        input_layout.addWidget(self.output_label)
        input_layout.addWidget(btn_output)
        input_group.setLayout(input_layout)

        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        btn_start = QPushButton("Начать обработку")
        btn_start.clicked.connect(self.start_processing)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.cancel_processing)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setText("Здесь будет отображен результат")
        self.preview.setStyleSheet("border: 1px solid gray;")

        layout.addWidget(input_group)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        layout.addWidget(btn_start)
        layout.addWidget(btn_cancel)
        layout.addWidget(self.preview)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.input_folder = folder
            self.input_label.setText(f"Папка с изображениями: {folder}")
            self.log_message(f"Выбрана папка: {folder}")

    def select_output_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", "", "JPEG (*.jpg);;PNG (*.png)")
        if filename:
            self.output_path = filename
            self.output_label.setText(f"Файл результата: {filename}")
            self.log_message(f"Результат будет сохранен в: {filename}")

    def log_message(self, message):
        self.log.append(message)

    def start_processing(self):
        if not hasattr(self, 'input_folder'):
            QMessageBox.warning(self, "Ошибка", "Сначала выберите папку с изображениями")
            return

        if not hasattr(self, 'output_path'):
            self.output_path = os.path.join(self.input_folder, "orthophoto_result.jpg")
            self.output_label.setText(f"Файл результата: {self.output_path}")
            self.log_message(f"Автоматически выбрано сохранение в: {self.output_path}")

        self.log_message("\nНачало обработки...")
        self.progress.setValue(0)

        self.processor = OrthoProcessor(self.input_folder, self.output_path)
        self.processor.progress_signal.connect(self.update_progress)
        self.processor.finished_signal.connect(self.process_finished)
        self.processor.error_signal.connect(self.process_error)
        self.processor.start()

    def cancel_processing(self):
        if hasattr(self, 'processor') and self.processor.isRunning():
            self.processor.cancel_process = True
            self.log_message("Обработка прервана пользователем")
            self.progress.setValue(0)

    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.log_message(message)

    def process_finished(self, result):
        self.log_message("Обработка успешно завершена!")
        h, w = result.shape[:2]
        preview = cv2.resize(result, (800, int(800 * h / w)))
        cv2.imwrite("preview.jpg", preview)
        self.preview.setPixmap(QPixmap("preview.jpg").scaled(800, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_error(self, message):
        QMessageBox.critical(self, "Ошибка", message)
        self.log_message(f"ОШИБКА: {message}")
        self.progress.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OrthoApp()
    window.show()
    sys.exit(app.exec_())