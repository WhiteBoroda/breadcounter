import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import json


@dataclass
class CameraConfig:
    oven_id: int
    camera_ip: str
    login: str
    password: str
    oven_name: str
    workshop_name: str
    enterprise_name: str


class CoralTPUPool:
    """Пул Coral TPU для обработки множества камер"""

    def __init__(self, num_tpu_devices=1, model_path='bread_detector_edgetpu.tflite'):
        self.num_devices = num_tpu_devices
        self.model_path = model_path
        self.detectors = []
        self.task_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=500)
        self.running = False

        # Инициализация TPU устройств
        self._initialize_tpu_devices()

        # Пул воркеров для обработки
        self.executor = ThreadPoolExecutor(max_workers=num_tpu_devices)

        logging.info(f"Initialized TPU pool with {num_tpu_devices} devices")

    def _initialize_tpu_devices(self):
        """Инициализация TPU устройств"""
        for i in range(self.num_devices):
            try:
                detector = CoralBreadDetector(
                    model_path=self.model_path,
                    device_path=f'/dev/apex_{i}' if i > 0 else None
                )
                self.detectors.append(detector)
            except Exception as e:
                logging.error(f"Failed to initialize TPU device {i}: {e}")

    def start(self):
        """Запуск пула обработки"""
        self.running = True
        for i, detector in enumerate(self.detectors):
            self.executor.submit(self._worker_loop, detector, i)

    def _worker_loop(self, detector, worker_id):
        """Основной цикл воркера TPU"""
        logging.info(f"TPU Worker {worker_id} started")

        while self.running:
            try:
                # Получаем задачу из очереди
                task = self.task_queue.get(timeout=1.0)
                oven_id, timestamp, frame = task

                # Обрабатываем кадр
                start_time = time.time()
                detections = detector.detect(frame)
                processing_time = time.time() - start_time

                # Отправляем результат
                result = {
                    'oven_id': oven_id,
                    'timestamp': timestamp,
                    'detections': detections,
                    'processing_time': processing_time,
                    'worker_id': worker_id
                }
                self.result_queue.put(result)

                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TPU Worker {worker_id} error: {e}")

    def submit_frame(self, oven_id: int, timestamp: float, frame) -> bool:
        """Отправка кадра на обработку"""
        try:
            self.task_queue.put((oven_id, timestamp, frame), timeout=0.1)
            return True
        except queue.Full:
            logging.warning(f"TPU queue full, dropping frame from oven {oven_id}")
            return False

    def get_result(self, timeout=0.1):
        """Получение результата обработки"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self):
        """Статистика пула TPU"""
        return {
            'active_devices': len(self.detectors),
            'queue_size': self.task_queue.qsize(),
            'results_pending': self.result_queue.qsize(),
            'total_capacity': self.task_queue.maxsize
        }

    def stop(self):
        """Остановка пула"""
        self.running = False
        self.executor.shutdown(wait=True)


class MultiCameraManager:
    """Менеджер множественных камер"""

    def __init__(self, tpu_pool: CoralTPUPool, db_session):
        self.tpu_pool = tpu_pool
        self.db_session = db_session

        # Камеры и их процессоры
        self.cameras: Dict[int, CameraProcessor] = {}
        self.trackers: Dict[int, BreadTracker] = {}
        self.counters: Dict[int, ProductionCounter] = {}

        # Статистика
        self.stats = defaultdict(lambda: {
            'frames_processed': 0,
            'detections_count': 0,
            'current_fps': 0,
            'last_activity': 0
        })

        self.running = False

    def add_camera(self, camera_config: CameraConfig):
        """Добавление камеры в систему"""
        oven_id = camera_config.oven_id

        # Создаем процессор камеры (без TPU - только захват кадров)
        camera = SimpleCameraCapture(
            camera_ip=camera_config.camera_ip,
            login=camera_config.login,
            password=camera_config.password,
            oven_id=oven_id
        )

        # Создаем трекер для этой камеры
        tracker = BreadTracker(oven_id=oven_id)

        # Создаем счетчик производства
        counter = ProductionCounter(self.db_session, oven_id)

        self.cameras[oven_id] = camera
        self.trackers[oven_id] = tracker
        self.counters[oven_id] = counter

        logging.info(f"Added camera for oven {oven_id}: {camera_config.oven_name}")

    def start_all_cameras(self):
        """Запуск всех камер"""
        self.running = True

        # Запускаем захват с каждой камеры
        for oven_id, camera in self.cameras.items():
            if camera.connect():
                threading.Thread(
                    target=self._camera_capture_loop,
                    args=(oven_id, camera),
                    daemon=True
                ).start()
                logging.info(f"Started capture for oven {oven_id}")
            else:
                logging.error(f"Failed to connect camera for oven {oven_id}")

        # Запускаем обработку результатов TPU
        threading.Thread(target=self._process_tpu_results, daemon=True).start()

        # Запускаем мониторинг статистики
        threading.Thread(target=self._stats_monitor, daemon=True).start()

    def _camera_capture_loop(self, oven_id: int, camera):
        """Цикл захвата кадров с камеры"""
        fps_counter = 0
        last_fps_time = time.time()

        while self.running:
            frame = camera.get_frame()
            if frame is not None:
                timestamp = time.time()

                # Отправляем кадр в TPU пул
                if self.tpu_pool.submit_frame(oven_id, timestamp, frame):
                    fps_counter += 1

                # Обновляем FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.stats[oven_id]['current_fps'] = fps_counter
                    fps_counter = 0
                    last_fps_time = current_time

                self.stats[oven_id]['last_activity'] = timestamp

            time.sleep(0.033)  # ~30 FPS

    def _process_tpu_results(self):
        """Обработка результатов от TPU пула"""
        while self.running:
            result = self.tpu_pool.get_result(timeout=0.1)
            if result:
                oven_id = result['oven_id']
                detections = result['detections']
                timestamp = result['timestamp']

                # Обновляем трекер для этой печи
                if oven_id in self.trackers:
                    # Фильтруем детекции
                    bread_detections = [d for d in detections if d['class_name'] in ['bread', 'loaf']]
                    marker_detections = [d for d in detections if d['class_name'] in ['circle', 'square', 'triangle']]

                    # Обновляем трекер
                    tracked_objects = self.trackers[oven_id].update(bread_detections)

                    # Обрабатываем результаты счетчиком
                    if oven_id in self.counters:
                        self.counters[oven_id].process_detections(
                            bread_detections,
                            marker_detections,
                            tracked_objects,
                            timestamp
                        )

                    # Обновляем статистику
                    self.stats[oven_id]['frames_processed'] += 1
                    self.stats[oven_id]['detections_count'] += len(bread_detections)

    def _stats_monitor(self):
        """Мониторинг статистики системы"""
        while self.running:
            time.sleep(10)  # каждые 10 секунд
            self._print_system_stats()

    def _print_system_stats(self):
        """Вывод статистики системы"""
        print("\n" + "=" * 80)
        print("📊 СТАТИСТИКА МНОГОКАМЕРНОЙ СИСТЕМЫ")
        print("=" * 80)

        # Статистика TPU пула
        tpu_stats = self.tpu_pool.get_stats()
        print(f"🧠 TPU Pool: {tpu_stats['active_devices']} устройств, "
              f"очередь: {tpu_stats['queue_size']}/{tpu_stats['total_capacity']}")

        # Статистика по каждой печи
        for oven_id in sorted(self.cameras.keys()):
            stats = self.stats[oven_id]
            counter = self.counters.get(oven_id)

            # Информация о печи
            oven = self.db_session.query(Oven).filter_by(id=oven_id).first()
            oven_name = oven.name if oven else f"Печь {oven_id}"

            print(f"\n🔥 {oven_name}:")
            print(f"   📹 FPS: {stats['current_fps']:2d} | "
                  f"Кадров: {stats['frames_processed']:5d} | "
                  f"Детекций: {stats['detections_count']:4d}")

            if counter and counter.current_batch:
                product_name = counter.current_product.name if counter.current_product else "Неизвестно"
                print(f"   🥖 Текущая партия: {product_name} - {counter.current_batch.total_count} шт")
            else:
                print(f"   ⏸️  Партия не активна")

            # Проверка активности камеры
            last_activity = stats['last_activity']
            if last_activity > 0:
                inactive_time = time.time() - last_activity
                if inactive_time > 30:
                    print(f"   ⚠️  Камера неактивна {inactive_time:.1f} сек")

    def get_oven_status(self, oven_id: int) -> dict:
        """Получение статуса конкретной печи"""
        if oven_id not in self.cameras:
            return {'error': 'Oven not found'}

        stats = self.stats[oven_id]
        counter = self.counters.get(oven_id)
        tracker = self.trackers.get(oven_id)

        status = {
            'oven_id': oven_id,
            'fps': stats['current_fps'],
            'frames_processed': stats['frames_processed'],
            'detections_count': stats['detections_count'],
            'last_activity': stats['last_activity'],
            'tracked_objects': len(tracker.objects) if tracker else 0
        }

        if counter and counter.current_batch:
            status['current_batch'] = {
                'product_name': counter.current_product.name,
                'start_time': counter.current_batch.start_time.isoformat(),
                'count': counter.current_batch.total_count,
                'defects': counter.current_batch.defect_count
            }

        return status

    def get_system_overview(self) -> dict:
        """Общий обзор системы"""
        active_cameras = sum(1 for stats in self.stats.values()
                             if time.time() - stats['last_activity'] < 60)

        active_batches = sum(1 for counter in self.counters.values()
                             if counter.current_batch is not None)

        total_processed = sum(stats['frames_processed'] for stats in self.stats.values())
        total_detections = sum(stats['detections_count'] for stats in self.stats.values())

        return {
            'total_cameras': len(self.cameras),
            'active_cameras': active_cameras,
            'active_batches': active_batches,
            'total_frames_processed': total_processed,
            'total_detections': total_detections,
            'tpu_stats': self.tpu_pool.get_stats()
        }

    def stop_all(self):
        """Остановка всех камер"""
        self.running = False

        # Завершаем активные партии
        for counter in self.counters.values():
            if counter.current_batch:
                counter.finish_current_batch()

        # Останавливаем камеры
        for camera in self.cameras.values():
            camera.disconnect()

        logging.info("All cameras stopped")


class SimpleCameraCapture:
    """Простой захват кадров с IP камеры"""

    def __init__(self, camera_ip: str, login: str, password: str, oven_id: int):
        self.camera_ip = camera_ip
        self.login = login
        self.password = password
        self.oven_id = oven_id
        self.cap = None

    def connect(self) -> bool:
        """Подключение к камере"""
        rtsp_url = f"rtsp://{self.login}:{self.password}@{self.camera_ip}/stream1"
        self.cap = cv2.VideoCapture(rtsp_url)

        if self.cap.isOpened():
            # Настройка параметров
            self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            return True
        return False

    def get_frame(self):
        """Получение кадра"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def disconnect(self):
        """Отключение от камеры"""
        if self.cap:
            self.cap.release()