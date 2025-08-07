# detection/coral_bread_detector.py
"""Детектор хлеба на Coral TPU с интегрированным трекингом"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass

# Coral TPU импорты
try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common, detect
    import tflite_runtime.interpreter as tflite

    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False
    print("⚠️  Coral TPU недоступен, используем CPU")

from core.tpu_manager import TPUManager


@dataclass
class TrackedBread:
    """Отслеживаемый объект хлеба"""
    id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    confidence: float
    frames_alive: int = 1
    last_seen: float = 0
    counted: bool = False
    zone: str = "unknown"


class CoralBreadDetector:
    """Детектор хлеба на Coral TPU с ByteTracker интеграцией"""

    def __init__(self, use_coral=True):
        self.use_coral = use_coral and CORAL_AVAILABLE
        self.tpu_manager = TPUManager()
        self.interpreter = None

        # Трекинг параметры
        self.tracks: Dict[int, TrackedBread] = {}
        self.next_id = 1
        self.frame_count = 0
        self.total_count = 0
        self.counted_ids = set()

        # Настройки детекции
        self.conf_threshold = 0.4
        self.iou_threshold = 0.3
        self.max_lost_frames = 15

        # Зоны подсчета
        self.counting_line_x = 700  # Позиция линии подсчета
        self.bread_width_range = (50, 200)  # Ожидаемая ширина хлеба
        self.bread_height_range = (30, 120)  # Ожидаемая высота хлеба

        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели детекции"""
        if self.use_coral:
            try:
                print("🔍 Диагностика Coral TPU...")
                self._diagnose_coral()

                # Ищем СУЩЕСТВУЮЩИЕ модели в папке models
                existing_models = [
                    'models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                    'models/mobilenet_v1_1.0_224_quant_edgetpu.tflite'  # Твоя существующая модель
                ]

                model_found = False
                for model_path in existing_models:
                    if os.path.exists(model_path):
                        try:
                            print(f"🔄 Загружаем существующую модель: {model_path}")

                            # Создаем интерпретатор напрямую
                            from pycoral.utils import edgetpu
                            from pycoral.adapters import common

                            self.interpreter = edgetpu.make_interpreter(model_path)
                            self.interpreter.allocate_tensors()

                            self.input_details = self.interpreter.get_input_details()
                            self.output_details = self.interpreter.get_output_details()

                            print(f"✅ TPU модель загружена: {os.path.basename(model_path)}")
                            print(f"📊 Входной размер: {self.input_details[0]['shape']}")

                            # Определяем тип модели
                            if 'ssd' in model_path.lower():
                                self.model_type = 'object_detection'
                                print("🎯 Тип: Object Detection (SSD)")
                            else:
                                self.model_type = 'classification'
                                print("🏷️  Тип: Classification (будет использоваться с CPU детекцией)")

                            # Тестовый inference
                            self._test_inference()

                            print("🎉 TPU РАБОТАЕТ!")
                            model_found = True
                            break

                        except Exception as e:
                            print(f"❌ Ошибка загрузки {model_path}: {e}")
                            continue

                if not model_found:
                    print("❌ Не найдено подходящих TPU моделей")
                    print("💡 Рекомендация: скачайте SSD MobileNet модель")
                    print(
                        "   wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
                    self.use_coral = False

            except Exception as e:
                print(f"❌ Критическая ошибка TPU: {e}")
                self.use_coral = False

        if not self.use_coral:
            print("🔄 Используется CPU детекция (высокое качество, средняя скорость)")

    def _diagnose_coral(self):
        """Диагностика состояния Coral TPU (PCIe версия)"""
        try:
            import subprocess

            # Проверка библиотек
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if 'libedgetpu.so.1' in result.stdout:
                print("✅ libedgetpu.so.1 найдена")
            else:
                print("❌ libedgetpu.so.1 не найдена")

            # Проверка PCIe устройства
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            if 'Global Unichip Corp. Coral Edge TPU' in result.stdout:
                print("✅ Coral PCIe TPU обнаружен в PCI")
            else:
                print("❌ Coral PCIe устройство не найдено в lspci")

            # Проверка модуля ядра apex
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'apex' in result.stdout:
                print("✅ Модуль ядра apex загружен")
            else:
                print("❌ Модуль ядра apex не загружен - выполните: sudo modprobe apex")

            # Проверка устройства /dev/apex_0
            if os.path.exists('/dev/apex_0'):
                print("✅ Устройство /dev/apex_0 найдено")
                # Проверяем права доступа
                import stat
                st = os.stat('/dev/apex_0')
                print(f"📋 Права доступа: {stat.filemode(st.st_mode)}")
            else:
                print("❌ Устройство /dev/apex_0 не найдено")

            # Проверка устройств Edge TPU через API
            devices = self.tpu_manager.list_devices()
            print(f"📊 Найдено Edge TPU устройств: {len(devices)}")
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device}")

        except Exception as e:
            print(f"⚠️  Не удалось выполнить диагностику: {e}")

    def _create_test_model(self) -> Optional[str]:
        """Создание/загрузка правильной object detection модели для TPU"""
        try:
            os.makedirs('models', exist_ok=True)

            # Правильная object detection модель
            detection_model_url = "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
            detection_model_path = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"

            # Файл меток
            labels_url = "https://github.com/google-coral/test_data/raw/master/coco_labels.txt"
            labels_path = "models/coco_labels.txt"

            # Скачиваем модель если её нет
            if not os.path.exists(detection_model_path):
                print(f"📥 Скачиваем object detection модель для TPU...")
                import urllib.request
                urllib.request.urlretrieve(detection_model_url, detection_model_path)
                print(f"✅ Модель скачана: {detection_model_path}")

            # Скачиваем метки если их нет
            if not os.path.exists(labels_path):
                print(f"📥 Скачиваем метки классов...")
                import urllib.request
                urllib.request.urlretrieve(labels_url, labels_path)
                print(f"✅ Метки скачаны: {labels_path}")

            return detection_model_path if os.path.exists(detection_model_path) else None

        except Exception as e:
            print(f"❌ Не удалось загрузить detection модель: {e}")

            # Fallback на классификационную модель
            try:
                fallback_url = "https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite"
                fallback_path = "models/mobilenet_v1_1.0_224_quant_edgetpu.tflite"

                if not os.path.exists(fallback_path):
                    import urllib.request
                    urllib.request.urlretrieve(fallback_url, fallback_path)
                    print(f"✅ Fallback модель загружена: {fallback_path}")

                return fallback_path if os.path.exists(fallback_path) else None

            except Exception as e2:
                print(f"❌ Ошибка загрузки fallback модели: {e2}")
                return None

    def _test_inference(self):
        """Тестовый inference для проверки работы TPU"""
        try:
            input_shape = self.input_details[0]['shape']
            test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)

            common.set_input(self.interpreter, test_input)
            self.interpreter.invoke()

            print("✅ Тестовый inference выполнен успешно")

        except Exception as e:
            print(f"❌ Ошибка тестового inference: {e}")
            raise

    def _detect_with_coral(self, frame: np.ndarray) -> List[Dict]:
        """Детекция с использованием Coral TPU (SSD MobileNet)"""
        if not self.interpreter:
            return []

        try:
            # Подготовка изображения для SSD MobileNet (300x300)
            input_shape = self.input_details[0]['shape'][1:3]  # [300, 300]
            resized_frame = cv2.resize(frame, input_shape)

            if len(resized_frame.shape) == 3:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Установка входных данных
            from pycoral.adapters import common, detect
            common.set_input(self.interpreter, resized_frame)

            # Инференс
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000

            # Получение результатов детекции
            objects = detect.get_objects(self.interpreter, self.conf_threshold, (1.0, 1.0))

            # Масштабирование координат к оригинальному размеру
            height, width = frame.shape[:2]
            scale_x = width / input_shape[1]
            scale_y = height / input_shape[0]

            detections = []
            for i, obj in enumerate(objects):
                # Масштабируем bounding box
                bbox = obj.bbox.scale(scale_x, scale_y)
                x, y, w, h = int(bbox.xmin), int(bbox.ymin), int(bbox.width), int(bbox.height)

                # Фильтр по размеру (приблизительно размер хлеба)
                if (self.bread_width_range[0] <= w <= self.bread_width_range[1] and
                        self.bread_height_range[0] <= h <= self.bread_height_range[1]):
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'confidence': obj.score,
                        'area': w * h,
                        'class_id': obj.id,
                        'inference_time': inference_time
                    })

            if detections:
                print(f"🚀 TPU обнаружил {len(detections)} объектов за {inference_time:.1f}ms")

            return detections

        except Exception as e:
            print(f"❌ Ошибка TPU детекции: {e}")
            # Fallback на простую детекцию
            return self._detect_simple(frame)

    def _detect_simple(self, frame: np.ndarray) -> List[Dict]:
        """Простая детекция по цвету (fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Расширенный диапазон для хлеба
        lower_bread1 = np.array([8, 30, 50])
        upper_bread1 = np.array([25, 255, 220])

        lower_bread2 = np.array([15, 20, 80])
        upper_bread2 = np.array([35, 180, 255])

        mask1 = cv2.inRange(hsv, lower_bread1, upper_bread1)
        mask2 = cv2.inRange(hsv, lower_bread2, upper_bread2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Морфологическая обработка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Найти контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if 1500 < area < 15000:  # Фильтр по площади
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Фильтр по соотношению сторон и размеру
                if (0.8 < aspect_ratio < 3.0 and
                        self.bread_width_range[0] <= w <= self.bread_width_range[1] and
                        self.bread_height_range[0] <= h <= self.bread_height_range[1]):
                    # Оценка уверенности на основе характеристик контура
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                    confidence = min(0.9, 0.3 + circularity * 0.5 + min(area / 8000, 0.4))

                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'confidence': confidence,
                        'area': int(area),
                        'class_id': 0,
                        'inference_time': 0
                    })

        return detections

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Расчет IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update_tracking(self, detections: List[Dict]) -> List[TrackedBread]:
        """Обновление трекинга объектов"""
        self.frame_count += 1
        current_time = time.time()

        # Матрица IoU для ассоциации
        active_tracks = list(self.tracks.values())
        iou_matrix = np.zeros((len(active_tracks), len(detections)))

        for i, track in enumerate(active_tracks):
            for j, detection in enumerate(detections):
                iou = self.calculate_iou(track.bbox, detection['bbox'])
                iou_matrix[i, j] = iou

        # Простая жадная ассоциация
        matched_tracks = set()
        matched_detections = set()

        # Высокоуверенные совпадения
        for i in range(len(active_tracks)):
            best_match = -1
            best_iou = self.iou_threshold

            for j in range(len(detections)):
                if j in matched_detections:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_match = j

            if best_match >= 0:
                track = active_tracks[i]
                detection = detections[best_match]

                # Обновляем трек
                track.bbox = detection['bbox']
                track.center = detection['center']
                track.confidence = detection['confidence']
                track.frames_alive += 1
                track.last_seen = current_time

                matched_tracks.add(i)
                matched_detections.add(best_match)

        # Создаем новые треки
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedBread(
                    id=self.next_id,
                    bbox=detection['bbox'],
                    center=detection['center'],
                    confidence=detection['confidence'],
                    last_seen=current_time
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Удаляем старые треки
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > self.max_lost_frames / 15:  # Примерно 1 сек при 15 FPS
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return list(self.tracks.values())

    def check_counting(self, tracks: List[TrackedBread], prev_positions: Dict[int, float]):
        """Проверка пересечения линии подсчета"""
        for track in tracks:
            if track.id in prev_positions and not track.counted:
                prev_x = prev_positions[track.id]
                curr_x = track.center[0]

                # Проверяем пересечение линии подсчета (движение слева направо)
                if prev_x < self.counting_line_x <= curr_x:
                    if track.id not in self.counted_ids:
                        self.total_count += 1
                        self.counted_ids.add(track.id)
                        track.counted = True
                        print(f"🍞 Подсчитан хлеб ID: {track.id}, Общий счет: {self.total_count}")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], int]:
        """Основная обработка кадра"""
        # Детекция
        if self.use_coral:
            detections = self._detect_with_coral(frame)
        else:
            detections = self._detect_simple(frame)

        # Сохраняем предыдущие позиции для подсчета
        prev_positions = {track.id: track.center[0] for track in self.tracks.values()}

        # Трекинг
        tracks = self.update_tracking(detections)

        # Подсчет
        self.check_counting(tracks, prev_positions)

        # Конвертируем для интерфейса
        interface_objects = []
        for track in tracks:
            interface_objects.append({
                'id': f"bread_{track.id}",
                'bbox': {
                    'x': track.bbox[0], 'y': track.bbox[1],
                    'width': track.bbox[2], 'height': track.bbox[3]
                },
                'center': {'x': track.center[0], 'y': track.center[1]},
                'confidence': track.confidence,
                'area': track.bbox[2] * track.bbox[3],
                'tracked_id': track.id,
                'frames_alive': track.frames_alive,
                'counted': track.counted,
                'zone': track.zone
            })

        return interface_objects, self.total_count

    def visualize_results(self, frame: np.ndarray, tracks: List[TrackedBread]) -> np.ndarray:
        """Визуализация результатов"""
        result = frame.copy()

        # Рисуем линию подсчета
        height = frame.shape[0]
        cv2.line(result, (self.counting_line_x, 0), (self.counting_line_x, height), (0, 255, 0), 3)
        cv2.putText(result, "COUNTING LINE", (self.counting_line_x - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Рисуем треки
        for track in tracks:
            x, y, w, h = track.bbox

            if track.counted:
                color = (0, 255, 0)  # Зеленый - подсчитан
                thickness = 3
            else:
                color = (255, 0, 0)  # Синий - активный трек
                thickness = 2

            # Bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

            # Информация о треке
            label = f"ID:{track.id}"
            if track.counted:
                label += " ✓"
            else:
                label += f" ({track.frames_alive}f)"

            cv2.putText(result, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Центр
            cv2.circle(result, (int(track.center[0]), int(track.center[1])), 4, color, -1)

        # Общий счетчик
        cv2.putText(result, f"TOTAL: {self.total_count}", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Статус TPU
        tpu_status = "TPU" if self.use_coral else "CPU"
        cv2.putText(result, f"Mode: {tpu_status}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return result

    def reset_counting(self):
        """Сброс счетчика"""
        self.total_count = 0
        self.counted_ids.clear()
        for track in self.tracks.values():
            track.counted = False

    def get_statistics(self) -> Dict:
        """Получение статистики"""
        return {
            'total_count': self.total_count,
            'active_tracks': len(self.tracks),
            'using_coral': self.use_coral,
            'frame_count': self.frame_count,
            'avg_confidence': sum(t.confidence for t in self.tracks.values()) / len(self.tracks) if self.tracks else 0
        }