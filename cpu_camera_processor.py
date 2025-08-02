import cv2
import threading
import time
from collections import deque


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