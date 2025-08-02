# coral_detector.py - Детектор для Coral TPU
import numpy as np
import cv2
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
import tflite_runtime.interpreter as tflite
from PIL import Image
import time


class CoralBreadDetector:
    """Детектор хлеба на Coral TPU"""

    def __init__(self, model_path='bread_detector_edgetpu.tflite', labels_path='labels.txt'):
        print("🧠 Инициализация Coral TPU детектора...")

        try:
            # Инициализация Coral TPU
            self.interpreter = edgetpu.make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            print("✅ Coral TPU инициализирован")
        except Exception as e:
            print(f"❌ Ошибка инициализации Coral TPU: {e}")
            raise

        # Загрузка меток
        try:
            self.labels = dataset.read_label_file(labels_path) if labels_path else {}
            print(f"✅ Загружено {len(self.labels)} классов")
        except:
            print("⚠️  Файл меток не найден, используем базовые классы")
            self.labels = {0: 'background', 1: 'bread', 2: 'circle', 3: 'square', 4: 'triangle'}

        # Параметры модели
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = common.input_size(self.interpreter)

        # Статистика производительности
        self.inference_times = []

        print(f"🎯 Модель готова, размер входа: {self.input_size}")

    def detect(self, frame):
        """Детекция объектов на кадре"""
        start_time = time.time()

        # Предобработка изображения
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        resized_image = pil_image.resize(self.input_size, Image.LANCZOS)

        # Инференс на Coral TPU
        common.set_input(self.interpreter, resized_image)
        self.interpreter.invoke()

        # Извлечение результатов
        objs = detect.get_objects(self.interpreter, threshold=0.4)

        # Масштабирование координат к оригинальному размеру
        scale_x = frame.shape[1] / self.input_size[0]
        scale_y = frame.shape[0] / self.input_size[1]

        detections = []
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)

            detection = {
                'class_id': obj.id,
                'class_name': self.labels.get(obj.id, 'unknown'),
                'confidence': obj.score,
                'bbox': (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax),
                'center': ((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2)
            }
            detections.append(detection)

        # Статистика производительности
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        return detections

    def get_performance_stats(self):
        """Статистика производительности"""
        if not self.inference_times:
            return {'avg_time': 0, 'fps': 0, 'device': 'Coral TPU'}

        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times),
            'device': 'Coral TPU'
        }