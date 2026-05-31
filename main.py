import sys
import time
import re
import json
import os
from difflib import SequenceMatcher

import cv2
import mss
import numpy as np
import easyocr

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QLabel, QTextEdit,
    QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen


SETTINGS_FILE = "ocr_settings.json"


class TransparentBox(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("텍스트 인식 영역")
        self.setGeometry(300, 300, 500, 250)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.dragging = False
        self.resizing = False
        self.drag_pos = None
        self.resize_margin = 16

    def paintEvent(self, event):
        painter = QPainter(self)

        pen = QPen(QColor(0, 255, 0), 3)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))

        painter.fillRect(
            self.width() - self.resize_margin,
            self.height() - self.resize_margin,
            self.resize_margin,
            self.resize_margin,
            QColor(0, 255, 0, 180)
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            y = event.position().y()

            if (
                x >= self.width() - self.resize_margin
                and y >= self.height() - self.resize_margin
            ):
                self.resizing = True
            else:
                self.dragging = True
                self.drag_pos = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )

    def mouseMoveEvent(self, event):
        if self.resizing:
            global_pos = event.globalPosition().toPoint()
            top_left = self.frameGeometry().topLeft()

            self.resize(
                max(100, global_pos.x() - top_left.x()),
                max(50, global_pos.y() - top_left.y())
            )

        elif self.dragging:
            self.move(
                event.globalPosition().toPoint() - self.drag_pos
            )

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False

    def get_capture_rect(self):
        geo = self.geometry()
        border = 6

        return {
            "left": geo.x() + border,
            "top": geo.y() + border,
            "width": max(1, geo.width() - border * 2),
            "height": max(1, geo.height() - border * 2),
        }


class OCRWorker(QThread):
    text_signal = pyqtSignal(object)
    status_signal = pyqtSignal(str)

    def __init__(self, roi_provider):
        super().__init__()

        self.roi_provider = roi_provider
        self.running = True

        self.reader = easyocr.Reader(
            ['ko', 'en'],
            gpu=False
        )

    def preprocess(self, image_bgr):
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.convertScaleAbs(
            gray,
            alpha=1.4,
            beta=10
        )

        return gray

    def get_text_color(self, image_bgr, bbox):
        x1 = int(min(point[0] for point in bbox))
        y1 = int(min(point[1] for point in bbox))
        x2 = int(max(point[0] for point in bbox))
        y2 = int(max(point[1] for point in bbox))

        h, w = image_bgr.shape[:2]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return (0, 0, 0)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask = hsv[:, :, 2] > 80

        if np.count_nonzero(mask) == 0:
            mean_color = cv2.mean(crop)[:3]
        else:
            mean_color = cv2.mean(crop, mask=mask.astype(np.uint8))[:3]

        b = int(round(mean_color[0] / 20) * 20)
        g = int(round(mean_color[1] / 20) * 20)
        r = int(round(mean_color[2] / 20) * 20)

        return (
            max(0, min(255, b)),
            max(0, min(255, g)),
            max(0, min(255, r)),
        )

    def run(self):
        try:
            with mss.mss() as sct:
                while self.running:
                    try:
                        capture_rect = self.roi_provider()

                        screenshot = sct.grab(capture_rect)
                        img = np.array(screenshot)

                        image_bgr = cv2.cvtColor(
                            img,
                            cv2.COLOR_BGRA2BGR
                        )

                        processed = self.preprocess(image_bgr)

                        result = self.reader.readtext(
                            processed,
                            detail=1,
                            paragraph=False
                        )

                        ocr_results = []

                        for item in result:
                            bbox = item[0]
                            text = item[1]

                            if not text.strip():
                                continue

                            x = int(bbox[0][0])
                            y = int(bbox[0][1])

                            color = self.get_text_color(
                                image_bgr,
                                bbox
                            )

                            ocr_results.append({
                                "text": text.strip(),
                                "x": x,
                                "y": y,
                                "color": color,
                            })

                        if ocr_results:
                            self.text_signal.emit(ocr_results)
                            self.status_signal.emit("텍스트 인식됨")
                        else:
                            self.status_signal.emit("텍스트 없음")

                    except Exception as e:
                        self.status_signal.emit(f"OCR 오류: {e}")
                        print("OCR 오류:", e)

                    time.sleep(1)

        except Exception as e:
            self.status_signal.emit(f"스레드 오류: {e}")
            print("스레드 오류:", e)

    def stop(self):
        self.running = False


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EasyOCR 화면 텍스트 인식 + 계산표")
        self.resize(850, 820)

        self.overlay_box = TransparentBox()
        self.worker = None

        self.elapsed_seconds = 0
        self.last_hour_total = 0
        self.hourly_profits = []

        self.detected_cache = {}

        layout = QVBoxLayout()

        self.show_box_button = QPushButton("인식 영역 보이기")
        layout.addWidget(self.show_box_button)

        self.hide_box_button = QPushButton("인식 영역 숨기기")
        layout.addWidget(self.hide_box_button)

        self.start_button = QPushButton("시작")
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("끄기")
        layout.addWidget(self.stop_button)

        self.status_label = QLabel("대기 중")
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("유사도 스코어"))
        self.score_input = QLineEdit()
        self.score_input.setText("0.75")
        layout.addWidget(self.score_input)

        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("인식된 텍스트가 여기에 표시됩니다.")
        layout.addWidget(self.text_box)

        self.add_row_button = QPushButton("+ 행 추가")
        layout.addWidget(self.add_row_button)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "이름",
            "갯수",
            "금액",
            "최종"
        ])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.table.setColumnWidth(0, 250)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 120)

        layout.addWidget(self.table)

        self.total_label = QLabel("전체 최종 금액: 0")
        layout.addWidget(self.total_label)

        self.hourly_average_label = QLabel("1시간 평균 상재: 0")
        layout.addWidget(self.hourly_average_label)

        self.setLayout(layout)

        self.show_box_button.clicked.connect(self.overlay_box.show)
        self.hide_box_button.clicked.connect(self.overlay_box.hide)
        self.start_button.clicked.connect(self.start_ocr)
        self.stop_button.clicked.connect(self.stop_ocr)
        self.add_row_button.clicked.connect(self.add_row)
        self.table.itemChanged.connect(self.on_table_item_changed)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)

        self.load_settings()
        self.overlay_box.show()

    def add_row(self):
        self.table.blockSignals(True)

        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem("0"))
        self.table.setItem(row, 2, QTableWidgetItem("0"))
        self.table.setItem(row, 3, QTableWidgetItem("0"))

        self.table.blockSignals(False)

        self.calculate_all_total()

    def on_table_item_changed(self, item):
        row = item.row()
        column = item.column()

        if column in (1, 2):
            self.calculate_row(row)

    def calculate_row(self, row):
        try:
            count_item = self.table.item(row, 1)
            price_item = self.table.item(row, 2)
            total_item = self.table.item(row, 3)

            if count_item is None or price_item is None or total_item is None:
                return

            count = int(count_item.text().replace(",", ""))
            price = int(price_item.text().replace(",", ""))
            total = count * price

        except Exception:
            total = 0

        self.table.blockSignals(True)
        self.table.item(row, 3).setText(str(total))
        self.table.blockSignals(False)

        self.calculate_all_total()

    def calculate_all_total(self):
        all_total = 0

        for row in range(self.table.rowCount()):
            try:
                total = int(
                    self.table.item(row, 3).text().replace(",", "")
                )
                all_total += total
            except Exception:
                pass

        self.total_label.setText(
            f"전체 최종 금액: {all_total:,}"
        )

    def get_current_total(self):
        text = self.total_label.text()
        text = text.replace("전체 최종 금액:", "")
        text = text.replace(",", "")
        text = text.strip()

        try:
            return int(text)
        except ValueError:
            return 0

    def update_timer(self):
        self.elapsed_seconds += 1

        if self.elapsed_seconds % 3600 == 0:
            current_total = self.get_current_total()
            one_hour_profit = current_total - self.last_hour_total

            self.hourly_profits.append(one_hour_profit)
            self.last_hour_total = current_total

        self.update_hourly_average()

    def update_hourly_average(self):
        passed_hours = len(self.hourly_profits)

        if passed_hours == 0:
            self.hourly_average_label.setText(
                "1시간 평균 상재: 0"
            )
            return

        average = sum(self.hourly_profits) / passed_hours

        self.hourly_average_label.setText(
            f"1시간 평균 상재: {average:,.0f} ({passed_hours}시간 지났음)"
        )

    def normalize_text(self, text):
        text = text.replace(" ", "")
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        text = text.replace(",", "")
        text = text.replace(".", "")
        text = text.replace(":", "")
        text = text.replace(";", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.strip()

        return text

    def similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def get_score_threshold(self):
        try:
            return float(self.score_input.text())
        except Exception:
            return 0.75

    def is_excluded_color(self, color):
        exclude_colors = [
            (230, 0, 230),
            (230, 230, 0),
        ]

        tolerance = 25

        for ex_color in exclude_colors:
            if (
                abs(color[0] - ex_color[0]) <= tolerance
                and abs(color[1] - ex_color[1]) <= tolerance
                and abs(color[2] - ex_color[2]) <= tolerance
            ):
                return True

        return False

    def check_ocr_text_and_count(self, ocr_results):
        current_time = time.time()
        threshold = self.get_score_threshold()

        for item in ocr_results:
            raw_text = item["text"]
            color = item["color"]
            y = item["y"]

            if self.is_excluded_color(color):
                continue

            target_text = self.normalize_text(raw_text)

            if not target_text:
                continue

            for row in range(self.table.rowCount()):
                name_item = self.table.item(row, 0)

                if name_item is None:
                    continue

                name = name_item.text().strip()

                if not name:
                    continue

                normalized_name = self.normalize_text(name)

                score = self.similarity(
                    normalized_name,
                    target_text
                )

                if score >= threshold:
                    is_duplicate = False

                    for _, last_info in list(self.detected_cache.items()):
                        same_name = last_info["name"] == normalized_name

                        cached_color = last_info["color"]
                        same_color = (
                            abs(color[0] - cached_color[0]) <= 25
                            and abs(color[1] - cached_color[1]) <= 25
                            and abs(color[2] - cached_color[2]) <= 25
                        )

                        close_y = abs(y - last_info["y"]) <= 100
                        recent = current_time - last_info["time"] < 12

                        if same_name and same_color and close_y and recent:
                            is_duplicate = True
                            break

                    if is_duplicate:
                        continue

                    cache_key = f"{normalized_name}_{color}_{int(current_time)}"

                    self.detected_cache[cache_key] = {
                        "name": normalized_name,
                        "color": color,
                        "y": y,
                        "time": current_time,
                    }

                    self.increase_row_count(row)
                    break

        for cache_key, last_info in list(self.detected_cache.items()):
            if current_time - last_info["time"] > 15:
                del self.detected_cache[cache_key]

    def increase_row_count(self, row):
        count_item = self.table.item(row, 1)

        if count_item is None:
            return

        try:
            count = int(count_item.text().replace(",", ""))
        except ValueError:
            count = 0

        count += 1

        self.table.blockSignals(True)
        count_item.setText(str(count))
        self.table.blockSignals(False)

        self.calculate_row(row)

    def save_settings(self):
        rows = []

        for row in range(self.table.rowCount()):
            row_data = []

            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                row_data.append(item.text() if item else "")

            rows.append(row_data)

        settings = {
            "window_x": self.x(),
            "window_y": self.y(),
            "window_w": self.width(),
            "window_h": self.height(),

            "box_x": self.overlay_box.x(),
            "box_y": self.overlay_box.y(),
            "box_w": self.overlay_box.width(),
            "box_h": self.overlay_box.height(),

            "rows": rows,

            "column_widths": [
                self.table.columnWidth(i)
                for i in range(self.table.columnCount())
            ],

            "score_threshold": self.score_input.text(),
        }

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print("설정 저장 실패:", e)

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)

            self.setGeometry(
                settings.get("window_x", 100),
                settings.get("window_y", 100),
                settings.get("window_w", 850),
                settings.get("window_h", 820),
            )

            self.overlay_box.setGeometry(
                settings.get("box_x", 300),
                settings.get("box_y", 300),
                settings.get("box_w", 500),
                settings.get("box_h", 250),
            )

            self.score_input.setText(
                settings.get("score_threshold", "0.75")
            )

            rows = settings.get("rows", [])

            self.table.blockSignals(True)
            self.table.setRowCount(0)

            for row_data in rows:
                row = self.table.rowCount()
                self.table.insertRow(row)

                for col in range(4):
                    value = row_data[col] if col < len(row_data) else ""
                    self.table.setItem(row, col, QTableWidgetItem(value))

            self.table.blockSignals(False)

            column_widths = settings.get("column_widths", [])

            for i, width in enumerate(column_widths):
                self.table.setColumnWidth(i, width)

            self.calculate_all_total()

        except Exception as e:
            print("설정 불러오기 실패:", e)

    def start_ocr(self):
        self.stop_ocr()

        self.overlay_box.show()

        self.detected_cache.clear()

        self.worker = OCRWorker(
            self.overlay_box.get_capture_rect
        )

        self.worker.text_signal.connect(self.update_text)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()

        self.elapsed_seconds = 0
        self.last_hour_total = self.get_current_total()
        self.hourly_profits = []
        self.hourly_average_label.setText("1시간 평균 상재: 0")
        self.timer.start(1000)

        self.status_label.setText("인식 시작")

    def stop_ocr(self):
        if self.worker:
            self.worker.stop()
            self.worker.quit()
            self.worker.wait()
            self.worker = None

        self.timer.stop()
        self.overlay_box.show()
        self.status_label.setText("중지됨")

    def update_text(self, ocr_results):
        display_lines = []

        for item in ocr_results:
            raw_text = item["text"]
            color = item["color"]

            target_text = self.normalize_text(raw_text)

            best_score = 0
            best_name = ""

            for row in range(self.table.rowCount()):
                name_item = self.table.item(row, 0)

                if name_item is None:
                    continue

                name = name_item.text().strip()

                if not name:
                    continue

                normalized_name = self.normalize_text(name)

                score = self.similarity(
                    normalized_name,
                    target_text
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            display_lines.append(
                f"{raw_text} | 매칭: {best_name} | score={best_score:.2f} | color={color}"
            )

        self.text_box.setPlainText(
            "\n".join(display_lines)
        )

        self.check_ocr_text_and_count(ocr_results)

    def update_status(self, text):
        self.status_label.setText(text)

    def closeEvent(self, event):
        self.save_settings()
        self.stop_ocr()
        self.overlay_box.close()
        event.accept()


app = QApplication(sys.argv)

window = Window()
window.show()

sys.exit(app.exec())