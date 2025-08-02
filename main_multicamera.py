# main_multicamera.py - Главный файл многокамерной системы
import sys
import time
import threading
import signal
import logging
from datetime import datetime
from collections import defaultdict

# Импорты компонентов системы
from config_loader import ConfigLoader
from models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Импорты для детекции и обработки
try:
    from coral_detector import CoralBreadDetector

    CORAL_AVAILABLE = True
except ImportError:
    from cpu_detector import CPUBreadDetector

    CORAL_AVAILABLE = False
    print("⚠️  Coral TPU недоступен, используем CPU детекцию")

from bread_tracker import BreadTracker
from production_counter import ProductionCounter
from web_api import start_monitoring_api

import cv2
import queue
import concurrent.futures


class CoralTPUPool:
    """Пул Coral TPU устройств для обработки множества камер"""

    def __init__(self, num_devices=1, model_path='bread_detector_edgetpu.tflite'):
        self.num_devices = num_devices
        self.model_path = model_path
        self.detectors = []
        self.task_queue = queue.Queue(maxsize=200)
        self.result_queue = queue.Queue(maxsize=1000)
        self.running = False

        # Инициализация детекторов
        self._initialize_detectors()

        # Пул воркеров
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_devices)

        logging.info(f"🧠 Инициализирован TPU пул с {len(self.detectors)} устройствами")

    def _initialize_detectors(self):
        """Инициализация TPU детекторов"""
        for i in range(self.num_devices):
            try:
                if CORAL_AVAILABLE:
                    detector = CoralBreadDetector(
                        model_path=self.model_path,
                        labels_path='labels.txt'
                    )
                else:
                    detector = CPUBreadDetector()

                self.detectors.append(detector)
                logging.info(f"✅ TPU детектор {i} инициализирован")

            except Exception as e:
                logging.error(f"❌ Ошибка инициализации детектора {i}: {e}")

    def start(self):
        """Запуск пула обработки"""
        if not self.detectors:
            raise RuntimeError("Нет доступных детекторов")

        self.running = True

        # Запускаем воркеры
        for i, detector in enumerate(self.detectors):
            self.executor.submit(self._worker_loop, detector, i)

        logging.info("🚀 TPU пул запущен")

    def _worker_loop(self, detector, worker_id):
        """Основной цикл воркера"""
        logging.info(f"🔄 TPU Worker {worker_id} запущен")

        while self.running:
            try:
                # Получаем задачу
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
                logging.error(f"❌ Ошибка в TPU Worker {worker_id}: {e}")

    def submit_frame(self, oven_id, timestamp, frame):
        """Отправка кадра на обработку"""
        try:
            self.task_queue.put((oven_id, timestamp, frame), timeout=0.1)
            return True
        except queue.Full:
            logging.warning(f"⚠️  TPU очередь переполнена, кадр с печи {oven_id} пропущен")
            return False

    def get_result(self, timeout=0.1):
        """Получение результата обработки"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self):
        """Статистика пула"""
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
        logging.info("🛑 TPU пул остановлен")


class CameraCapture:
    """Захват кадров с IP камеры"""

    def __init__(self, camera_config):
        self.config = camera_config
        self.cap = None
        self.running = False
        self.last_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

    def connect(self):
        """Подключение к камере"""
        rtsp_paths = [
            f"rtsp://{self.config.login}:{self.config.password}@{self.config.camera_ip}/stream1",
            f"rtsp://{self.config.login}:{self.config.password}@{self.config.camera_ip}/stream0",
            f"rtsp://{self.config.login}:{self.config.password}@{self.config.camera_ip}/live"
        ]

        for rtsp_url in rtsp_paths:
            try:
                self.cap = cv2.VideoCapture(rtsp_url)
                self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)

                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.last_frame = frame
                        logging.info(f"✅ Камера {self.config.oven_name} подключена ({frame.shape[1]}x{frame.shape[0]})")
                        return True

                self.cap.release()

            except Exception as e:
                logging.error(f"❌ Ошибка подключения к {self.config.oven_name}: {e}")

        return False

    def start_capture(self, tpu_pool):
        """Запуск захвата кадров"""
        self.running = True

        def capture_loop():
            while self.running and self.cap:
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame
                    timestamp = time.time()

                    # Отправляем кадр в TPU пул
                    tpu_pool.submit_frame(self.config.oven_id, timestamp, frame)

                    # Обновляем статистику
                    self.frame_count += 1
                    self._update_fps()

                time.sleep(0.067)  # ~15 FPS

        threading.Thread(target=capture_loop, daemon=True).start()
        logging.info(f"🎬 Захват запущен для {self.config.oven_name}")

    def _update_fps(self):
        """Обновление FPS"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def disconnect(self):
        """Отключение от камеры"""
        self.running = False
        if self.cap:
            self.cap.release()


class MultiCameraSystem:
    """Главная система управления множественными камерами"""

    def __init__(self, config_file='cameras.yaml'):
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bread_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('MultiCameraSystem')

        # Загрузка конфигурации
        self.config = ConfigLoader(config_file)
        self.camera_configs = self.config.get_cameras()
        self.system_settings = self.config.get_system_settings()

        # Настройка базы данных
        self.engine = create_engine('sqlite:///bread_production.db')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()

        # Компоненты системы
        self.tpu_pool = None
        self.cameras = {}
        self.trackers = {}
        self.counters = {}

        # Статистика
        self.stats = defaultdict(lambda: {
            'frames_processed': 0,
            'detections_count': 0,
            'current_fps': 0,
            'last_activity': 0
        })

        # Флаг работы системы
        self.running = False

        # Обработчики сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"🏗️  Система инициализирована для {len(self.camera_configs)} камер")

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        self.logger.info(f"🛑 Получен сигнал {signum}, завершаем работу...")
        self.stop()
        sys.exit(0)

    def initialize_components(self):
        """Инициализация всех компонентов системы"""
        self.logger.info("⚙️  Инициализация компонентов...")

        # 1. Инициализация TPU пула
        try:
            num_tpu = self.system_settings.get('tpu_devices', 1)
            self.tpu_pool = CoralTPUPool(num_devices=num_tpu)
            self.tpu_pool.start()
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации TPU пула: {e}")
            return False

        # 2. Инициализация камер
        connected_cameras = 0
        for camera_config in self.camera_configs:
            camera = CameraCapture(camera_config)

            if camera.connect():
                self.cameras[camera_config.oven_id] = camera

                # Инициализация трекера
                tracker = BreadTracker(camera_config.oven_id)

                # Настройка зон подсчета на основе первого кадра
                if camera.last_frame is not None:
                    h, w = camera.last_frame.shape[:2]
                    tracker.setup_counting_zones(h, w)

                self.trackers[camera_config.oven_id] = tracker

                # Инициализация счетчика производства
                counter = ProductionCounter(self.db_session, camera_config.oven_id)
                self.counters[camera_config.oven_id] = counter

                connected_cameras += 1
                self.logger.info(f"✅ Инициализирована печь {camera_config.oven_id}: {camera_config.oven_name}")
            else:
                self.logger.error(f"❌ Не удалось подключиться к печи {camera_config.oven_id}")

        if connected_cameras == 0:
            self.logger.error("❌ Ни одна камера не подключилась")
            return False

        self.logger.info(f"✅ Инициализировано {connected_cameras}/{len(self.camera_configs)} камер")
        return True

    def start_processing(self):
        """Запуск обработки всех камер"""
        if not self.initialize_components():
            return False

        self.running = True

        # Запускаем захват с каждой камеры
        for oven_id, camera in self.cameras.items():
            camera.start_capture(self.tpu_pool)

        # Запускаем обработку результатов TPU
        threading.Thread(target=self._process_tpu_results, daemon=True).start()

        # Запускаем мониторинг статистики
        threading.Thread(target=self._stats_monitor, daemon=True).start()

        # Запускаем веб-API
        try:
            api, api_thread = start_monitoring_api(self, host='0.0.0.0', port=5000)
            self.logger.info("🌐 Веб-интерфейс запущен: http://localhost:5000")
        except Exception as e:
            self.logger.error(f"⚠️  Не удалось запустить веб-интерфейс: {e}")

        self.logger.info("🚀 Обработка всех камер запущена")
        return True

    def _process_tpu_results(self):
        """Обработка результатов от TPU пула"""
        self.logger.info("🔄 Запущена обработка результатов TPU")

        while self.running:
            result = self.tpu_pool.get_result(timeout=0.1)
            if result:
                oven_id = result['oven_id']
                detections = result['detections']
                timestamp = result['timestamp']

                # Разделяем детекции на хлеб и маркеры
                bread_detections = [d for d in detections
                                    if d['class_name'] in ['bread', 'loaf']]
                marker_detections = [d for d in detections
                                     if d['class_name'] in ['circle', 'square', 'triangle', 'diamond', 'star']]

                # Обновляем трекер
                if oven_id in self.trackers:
                    tracked_objects = self.trackers[oven_id].update(bread_detections)

                    # Обрабатываем результаты счетчиком производства
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
                self.stats[oven_id]['last_activity'] = timestamp

    def _stats_monitor(self):
        """Мониторинг и вывод статистики"""
        while self.running:
            time.sleep(30)  # каждые 30 секунд
            self._print_system_stats()

    def _print_system_stats(self):
        """Вывод статистики системы"""
        try:
            print("\n" + "=" * 80)
            print("📊 СТАТИСТИКА МНОГОКАМЕРНОЙ СИСТЕМЫ")
            print(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)

            # Статистика TPU
            if self.tpu_pool:
                tpu_stats = self.tpu_pool.get_stats()
                print(f"🧠 TPU: {tpu_stats['active_devices']} устройств, "
                      f"очередь: {tpu_stats['queue_size']}/{tpu_stats['total_capacity']}")

            # Статистика по каждой печи
            for oven_id in sorted(self.cameras.keys()):
                camera = self.cameras[oven_id]
                stats = self.stats[oven_id]
                counter = self.counters.get(oven_id)

                print(f"\n🔥 {camera.config.oven_name} (ID: {oven_id}):")
                print(f"   📹 Камера FPS: {camera.current_fps:2d} | "
                      f"Кадров: {camera.frame_count:5d} | "
                      f"Детекций: {stats['detections_count']:4d}")

                # Информация о текущей партии
                if counter:
                    batch_info = counter.get_current_batch_info()
                    if batch_info:
                        print(f"   🥖 Партия: {batch_info['product_name']} - "
                              f"{batch_info['total_count']} шт "
                              f"(брак: {batch_info['defect_count']}) "
                              f"[{batch_info['duration_minutes']} мин]")
                    else:
                        print(f"   ⏸️  Партия не активна")

                # Статистика трекера
                if oven_id in self.trackers:
                    track_stats = self.trackers[oven_id].get_count_stats()
                    print(f"   🎯 Трекинг: {track_stats.get('active_objects', 0)} объектов, "
                          f"подсчет: {track_stats.get('total', 0)}")

                # Проверка активности
                if stats['last_activity'] > 0:
                    inactive_time = time.time() - stats['last_activity']
                    if inactive_time > 60:
                        print(f"   ⚠️  Неактивна {inactive_time / 60:.1f} мин")

        except Exception as e:
            self.logger.error(f"Ошибка вывода статистики: {e}")

    def get_system_overview(self):
        """Получение обзора системы для API"""
        active_cameras = sum(1 for stats in self.stats.values()
                             if time.time() - stats.get('last_activity', 0) < 60)

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
            'tpu_stats': self.tpu_pool.get_stats() if self.tpu_pool else {}
        }

    def get_oven_status(self, oven_id):
        """Получение статуса конкретной печи"""
        if oven_id not in self.cameras:
            return {'error': 'Oven not found'}

        camera = self.cameras[oven_id]
        stats = self.stats[oven_id]
        counter = self.counters.get(oven_id)
        tracker = self.trackers.get(oven_id)

        status = {
            'oven_id': oven_id,
            'oven_name': camera.config.oven_name,
            'fps': camera.current_fps,
            'frames_processed': stats['frames_processed'],
            'detections_count': stats['detections_count'],
            'last_activity': stats['last_activity'],
            'tracked_objects': len(tracker.objects) if tracker else 0
        }

        # Информация о текущей партии
        if counter:
            batch_info = counter.get_current_batch_info()
            if batch_info:
                status['current_batch'] = batch_info

        return status

    def stop(self):
        """Остановка всей системы"""
        self.logger.info("🛑 Остановка системы...")

        self.running = False

        # Завершаем активные партии
        for counter in self.counters.values():
            if counter.current_batch:
                counter.force_finish_batch("System shutdown")

        # Останавливаем камеры
        for camera in self.cameras.values():
            camera.disconnect()

        # Останавливаем TPU пул
        if self.tpu_pool:
            self.tpu_pool.stop()

        # Закрываем БД
        self.db_session.close()

        self.logger.info("✅ Система остановлена")

    def run(self):
        """Главный цикл работы системы"""
        if not self.start_processing():
            self.logger.error("❌ Не удалось запустить обработку")
            return False

        try:
            self.logger.info("🎯 Система работает. Нажмите Ctrl+C для остановки")

            # Главный цикл - просто ждем
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("🛑 Получен сигнал остановки от пользователя")
        finally:
            self.stop()

        return True


def main():
    """Главная функция"""
    print("🥖 МНОГОКАМЕРНАЯ СИСТЕМА ПОДСЧЕТА ХЛЕБА")
    print("=" * 60)

    # Проверяем аргументы командной строки
    config_file = 'cameras.yaml'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print(f"📁 Конфигурация: {config_file}")

    try:
        # Создаем и запускаем систему
        system = MultiCameraSystem(config_file)

        print(f"🏭 Найдено {len(system.camera_configs)} камер в конфигурации")
        for cam in system.camera_configs:
            print(f"   🔥 {cam.oven_name} ({cam.camera_ip})")

        print("\n🚀 Запуск системы...")
        success = system.run()

        if success:
            print("✅ Система завершена успешно")
        else:
            print("❌ Система завершена с ошибками")
            sys.exit(1)

    except FileNotFoundError:
        print(f"❌ Файл конфигурации не найден: {config_file}")
        print("   Создайте файл cameras.yaml или укажите правильный путь")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()