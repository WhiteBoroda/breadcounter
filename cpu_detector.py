# cpu_detector.py - CPU детектор и система тестирования
import cv2
import numpy as np
import time
import threading
from collections import deque
from ultralytics import YOLO


class CPUBreadDetector:
    """Детектор хлеба на CPU для предварительного тестирования"""

    def __init__(self, model_path='yolov8n.pt'):
        print("🖥️  Инициализация CPU детектора...")

        # Используем предобученную YOLOv8 модель
        try:
            self.model = YOLO(model_path)
            print("✅ YOLOv8 модель загружена")
        except:
            print("⚠️  YOLOv8 не найдена, используем простую детекцию")
            self.model = None

        # Статистика производительности
        self.inference_times = []

        # Простые параметры для детекции хлеба
        self.bread_size_range = (8000, 35000)  # площадь в пикселях
        self.marker_size_range = (1000, 8000)  # площадь маркеров

    def detect(self, frame):
        """Детекция объектов на кадре"""
        start_time = time.time()

        if self.model:
            # Используем YOLOv8 если доступна
            results = self.model(frame, verbose=False)
            detections = self._process_yolo_results(results, frame)
        else:
            # Простая детекция по контурам
            detections = self._simple_contour_detection(frame)

        # Статистика производительности
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 50:
            self.inference_times.pop(0)

        return detections

    def _process_yolo_results(self, results, frame):
        """Обработка результатов YOLO"""
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    if conf > 0.3:  # Низкий порог для тестирования
                        detection = {
                            'class_id': cls,
                            'class_name': 'object',  # Общее название
                            'confidence': float(conf),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                        }
                        detections.append(detection)

        return detections

    def _simple_contour_detection(self, frame):
        """Простая детекция по контурам (fallback)"""
        detections = []

        # Конвертируем в HSV для лучшей детекции хлеба
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Маска для хлебных цветов (коричневый/золотистый)
        lower_bread = np.array([10, 50, 50])
        upper_bread = np.array([30, 255, 255])
        bread_mask = cv2.inRange(hsv, lower_bread, upper_bread)

        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        bread_mask = cv2.morphologyEx(bread_mask, cv2.MORPH_CLOSE, kernel)
        bread_mask = cv2.morphologyEx(bread_mask, cv2.MORPH_OPEN, kernel)

        # Поиск контуров
        contours, _ = cv2.findContours(bread_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Классификация по размеру
            if self.bread_size_range[0] < area < self.bread_size_range[1]:
                x, y, w, h = cv2.boundingRect(contour)

                detection = {
                    'class_id': 1,
                    'class_name': 'bread',
                    'confidence': 0.8,
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w / 2, y + h / 2)
                }
                detections.append(detection)

            elif self.marker_size_range[0] < area < self.marker_size_range[1]:
                # Простая классификация маркеров по форме
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                marker_type = 'unknown'
                if 0.8 < aspect_ratio < 1.2:
                    marker_type = 'square'
                elif aspect_ratio > 1.5:
                    marker_type = 'triangle'
                else:
                    marker_type = 'circle'

                detection = {
                    'class_id': 2,
                    'class_name': marker_type,
                    'confidence': 0.7,
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w / 2, y + h / 2)
                }
                detections.append(detection)

        return detections

    def get_performance_stats(self):
        """Статистика производительности"""
        if not self.inference_times:
            return {'avg_time': 0, 'fps': 0, 'device': 'CPU'}

        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
            'device': 'CPU'
        }


class CPUCameraProcessor:
    """Упрощенный процессор камер для CPU тестирования"""

    def __init__(self, camera_ip, login, password, oven_id):
        self.camera_ip = camera_ip
        self.login = login
        self.password = password
        self.oven_id = oven_id

        # CPU детектор
        self.detector = CPUBreadDetector()

        # Простой трекер
        self.simple_tracker = SimpleObjectTracker()

        # Потоки и очереди
        self.cap = None
        self.running = False
        self.detection_results = deque(maxlen=50)

        # Статистика
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

    def connect_camera(self):
        """Подключение к IP-камере"""
        rtsp_paths = [
            f"rtsp://{self.login}:{self.password}@{self.camera_ip}/stream1",
            f"rtsp://{self.login}:{self.password}@{self.camera_ip}/stream0",
            f"rtsp://{self.login}:{self.password}@{self.camera_ip}/live"
        ]

        for rtsp_url in rtsp_paths:
            print(f"🔌 Пробуем подключиться: {rtsp_url.replace(self.password, '***')}")
            self.cap = cv2.VideoCapture(rtsp_url)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 10)  # Низкий FPS для CPU

                ret, frame = self.cap.read()
                if ret:
                    print(f"✅ Подключение успешно! Разрешение: {frame.shape[1]}x{frame.shape[0]}")
                    return True

        print(f"❌ Не удалось подключиться к камере {self.camera_ip}")
        return False

    def start_processing(self):
        """Запуск обработки"""
        self.running = True

        # Один поток для захвата и обработки (упрощенно для CPU)
        process_thread = threading.Thread(target=self._process_loop, daemon=True)
        process_thread.start()

        print(f"🎬 CPU обработка запущена для печи {self.oven_id}")

    def _process_loop(self):
        """Основной цикл обработки"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.time()

                # Детекция (каждый 3-й кадр для экономии CPU)
                if self.fps_counter % 3 == 0:
                    detections = self.detector.detect(frame)

                    # Простое отслеживание
                    tracked_objects = self.simple_tracker.update(detections)

                    # Сохраняем результат
                    result = {
                        'timestamp': timestamp,
                        'detections': detections,
                        'tracked_objects': tracked_objects,
                        'performance': self.detector.get_performance_stats()
                    }

                    self.detection_results.append(result)

                # Обновляем FPS
                self._update_fps()

            time.sleep(0.1)  # 10 FPS для CPU

    def _update_fps(self):
        """Обновление FPS"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def get_latest_results(self):
        """Получение последних результатов"""
        if self.detection_results:
            return self.detection_results[-1]
        return None

    def stop_processing(self):
        """Остановка обработки"""
        self.running = False
        if self.cap:
            self.cap.release()


class SimpleObjectTracker:
    """Простой трекер объектов для CPU версии"""

    def __init__(self):
        self.objects = {}
        self.next_id = 0
        self.max_distance = 100

    def update(self, detections):
        """Обновление трекера"""
        if not detections:
            return self.objects

        # Простое сопоставление по расстоянию
        current_centers = [(d['center'][0], d['center'][1]) for d in detections]

        # Для простоты - каждую детекцию считаем новым объектом
        self.objects = {}
        for i, detection in enumerate(detections):
            self.objects[i] = {
                'center': detection['center'],
                'class_name': detection['class_name'],
                'confidence': detection['confidence']
            }

        return self.objects


class CPUTestSystem:
    """Упрощенная система для тестирования без TPU"""

    def __init__(self):
        self.cameras = {}
        self.running = False

    def add_camera(self, oven_id, camera_ip, login, password):
        """Добавление камеры"""
        processor = CPUCameraProcessor(camera_ip, login, password, oven_id)

        if processor.connect_camera():
            self.cameras[oven_id] = processor
            print(f"✅ Камера печи {oven_id} добавлена")
            return True
        else:
            print(f"❌ Не удалось добавить камеру печи {oven_id}")
            return False

    def start_testing(self):
        """Запуск тестирования"""
        print("🚀 Запуск CPU тестирования...")

        for processor in self.cameras.values():
            processor.start_processing()

        self.running = True

        try:
            self._monitoring_loop()
        except KeyboardInterrupt:
            print("\n🛑 Остановка тестирования...")
            self.stop()

    def _monitoring_loop(self):
        """Цикл мониторинга"""
        while self.running:
            print("\n" + "=" * 60)
            print("📊 СТАТИСТИКА CPU ТЕСТИРОВАНИЯ")
            print("=" * 60)

            for oven_id, processor in self.cameras.items():
                results = processor.get_latest_results()

                if results:
                    detections = results['detections']
                    performance = results['performance']

                    bread_count = len([d for d in detections if d['class_name'] == 'bread'])
                    marker_count = len([d for d in detections if d['class_name'] in ['circle', 'square', 'triangle']])

                    print(f"\n🔥 Печь {oven_id}:")
                    print(f"   📹 FPS: {processor.current_fps:2d}")
                    print(f"   🧠 Обработка: {performance['fps']:.1f} FPS")
                    print(f"   🥖 Хлеб: {bread_count} шт")
                    print(f"   🎯 Маркеры: {marker_count} шт")
                    print(f"   ⏱️  Время детекции: {performance['avg_inference_time'] * 1000:.1f}мс")
                else:
                    print(f"\n🔥 Печь {oven_id}: Нет данных")

            time.sleep(10)  # Обновление каждые 10 секунд

    def stop(self):
        """Остановка системы"""
        self.running = False
        for processor in self.cameras.values():
            processor.stop_processing()
        print("✅ CPU тестирование остановлено")