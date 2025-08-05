# core/tpu_manager.py
"""Менеджер Coral TPU устройств"""

from .imports import *
import time
import logging


class TPUManager:
    """Централизованный менеджер Coral TPU"""

    _instance = None
    _devices = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = logging.getLogger(__name__)

    @staticmethod
    def is_available():
        """Проверка доступности Coral TPU"""
        return CORAL_AVAILABLE

    def list_devices(self):
        """Получить список TPU устройств"""
        if not CORAL_AVAILABLE:
            return []

        if self._devices is None:
            try:
                self._devices = edgetpu.list_edge_tpus()
                self.logger.info(f"Найдено TPU устройств: {len(self._devices)}")
            except Exception as e:
                self.logger.error(f"Ошибка поиска TPU: {e}")
                self._devices = []

        return self._devices

    def get_device_count(self):
        """Количество доступных TPU устройств"""
        return len(self.list_devices())

    def create_interpreter(self, model_path, device_id=0):
        """Создание интерпретатора для TPU"""
        if not CORAL_AVAILABLE:
            raise RuntimeError("Coral TPU недоступен")

        devices = self.list_devices()
        if not devices:
            raise RuntimeError("TPU устройства не найдены")

        if device_id >= len(devices):
            device_id = 0

        try:
            interpreter = edgetpu.make_interpreter(model_path, device=devices[device_id]['path'])
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            self.logger.error(f"Ошибка создания интерпретатора: {e}")
            raise

    def test_performance(self, model_path='bread_detector_edgetpu.tflite', iterations=100):
        """Тест производительности TPU"""
        if not CORAL_AVAILABLE:
            return None

        try:
            interpreter = self.create_interpreter(model_path)
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape']

            # Создаем тестовые данные
            test_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)

            # Прогрев
            for _ in range(10):
                common.set_input(interpreter, test_data)
                interpreter.invoke()

            # Измерение
            start_time = time.time()
            for _ in range(iterations):
                common.set_input(interpreter, test_data)
                interpreter.invoke()

            total_time = time.time() - start_time
            avg_time = (total_time / iterations) * 1000  # мс
            fps = iterations / total_time

            return {
                'avg_inference_ms': round(avg_time, 2),
                'fps': round(fps, 1),
                'total_time_s': round(total_time, 2)
            }

        except Exception as e:
            self.logger.error(f"Ошибка теста производительности: {e}")
            return None


class TPUPool:
    """Пул TPU устройств для многопоточной обработки"""

    def __init__(self, model_path, labels_path=None, max_workers=None):
        self.manager = TPUManager()
        self.model_path = model_path
        self.labels_path = labels_path

        device_count = self.manager.get_device_count()
        if device_count == 0:
            raise RuntimeError("TPU устройства не найдены")

        self.max_workers = min(max_workers or device_count, device_count)
        self.detectors = []
        self.task_queue = queue.Queue(maxsize=200)
        self.result_queue = queue.Queue(maxsize=1000)
        self.running = False

        self._initialize_detectors()

    def _initialize_detectors(self):
        """Инициализация детекторов для каждого TPU"""
        for i in range(self.max_workers):
            try:
                detector = self._create_detector(i)
                self.detectors.append(detector)
            except Exception as e:
                logging.error(f"Ошибка инициализации детектора {i}: {e}")

        if not self.detectors:
            raise RuntimeError("Не удалось создать ни одного детектора")

    def _create_detector(self, device_id):
        """Создание детектора для конкретного TPU"""
        # Динамический импорт для избежания циклических зависимостей
        from detection.coral_detector import CoralBreadDetector
        return CoralBreadDetector(
            model_path=self.model_path,
            labels_path=self.labels_path,
            device_id=device_id
        )

    def start(self):
        """Запуск пула"""
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

    def stop(self):
        """Остановка пула"""
        self.running = False
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    def submit_frame(self, frame, camera_id=None):
        """Отправка кадра на обработку"""
        if not self.running:
            return None

        try:
            self.task_queue.put((frame, camera_id), timeout=1)
            return True
        except queue.Full:
            return False

    def get_result(self, timeout=1):
        """Получение результата обработки"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None