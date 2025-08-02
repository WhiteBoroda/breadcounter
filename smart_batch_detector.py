# smart_batch_detector.py - Умное определение смены партий без маркеров
import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging


@dataclass
class BreadCharacteristics:
    """Характеристики хлебобулочного изделия"""
    avg_color: tuple  # Средний цвет (BGR)
    avg_size: float  # Средняя площадь
    aspect_ratio: float  # Соотношение сторон
    shape_complexity: float  # Сложность формы
    texture_variance: float  # Вариация текстуры


class BreadTypeClassifier:
    """Классификатор типов хлеба по визуальным признакам"""

    def __init__(self):
        # Эталонные характеристики типов хлеба
        self.bread_types = {
            'white_bread': {
                'color_range': [(180, 150, 120), (220, 200, 170)],  # Светлый
                'size_range': (8000, 15000),
                'aspect_ratio_range': (0.7, 1.3),
                'name': 'Белый хлеб'
            },
            'dark_bread': {
                'color_range': [(80, 60, 40), (140, 100, 80)],  # Темный
                'size_range': (8000, 15000),
                'aspect_ratio_range': (0.7, 1.3),
                'name': 'Черный хлеб'
            },
            'baton': {
                'color_range': [(160, 130, 100), (200, 170, 140)],  # Золотистый
                'size_range': (12000, 25000),
                'aspect_ratio_range': (1.5, 3.0),  # Продолговатый
                'name': 'Батон'
            },
            'rolls': {
                'color_range': [(170, 140, 110), (210, 180, 150)],  # Румяный
                'size_range': (3000, 8000),  # Маленькие
                'aspect_ratio_range': (0.8, 1.2),
                'name': 'Булочки'
            }
        }

        self.logger = logging.getLogger('BreadTypeClassifier')

    def extract_characteristics(self, frame, bbox) -> BreadCharacteristics:
        """Извлечение характеристик хлеба из области"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return BreadCharacteristics((0, 0, 0), 0, 1.0, 0, 0)

        # Средний цвет
        avg_color = tuple(np.mean(roi, axis=(0, 1)).astype(int))

        # Размер
        area = (x2 - x1) * (y2 - y1)

        # Соотношение сторон
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0

        # Сложность формы (упрощенная)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_complexity = len(contours[0]) if contours else 0

        # Вариация текстуры
        texture_variance = np.var(gray) if gray.size > 0 else 0

        return BreadCharacteristics(
            avg_color=avg_color,
            avg_size=area,
            aspect_ratio=aspect_ratio,
            shape_complexity=shape_complexity,
            texture_variance=texture_variance
        )

    def classify_bread_type(self, characteristics: BreadCharacteristics) -> str:
        """Классификация типа хлеба по характеристикам"""
        best_match = 'unknown'
        best_score = 0

        for bread_type, params in self.bread_types.items():
            score = 0

            # Проверка цвета
            color_min, color_max = params['color_range']
            if (color_min[0] <= characteristics.avg_color[0] <= color_max[0] and
                    color_min[1] <= characteristics.avg_color[1] <= color_max[1] and
                    color_min[2] <= characteristics.avg_color[2] <= color_max[2]):
                score += 3

            # Проверка размера
            size_min, size_max = params['size_range']
            if size_min <= characteristics.avg_size <= size_max:
                score += 2

            # Проверка соотношения сторон
            ratio_min, ratio_max = params['aspect_ratio_range']
            if ratio_min <= characteristics.aspect_ratio <= ratio_max:
                score += 2

            if score > best_score:
                best_score = score
                best_match = bread_type

        return best_match if best_score >= 4 else 'unknown'

    def get_bread_name(self, bread_type: str) -> str:
        """Получение читаемого названия типа хлеба"""
        return self.bread_types.get(bread_type, {}).get('name', 'Неизвестный тип')


class GapDetector:
    """Детектор пропусков между партиями"""

    def __init__(self, min_gap_rows=2, max_gap_rows=50):
        self.min_gap_rows = min_gap_rows
        self.max_gap_rows = max_gap_rows
        self.gap_history = deque(maxlen=100)
        self.logger = logging.getLogger('GapDetector')

    def detect_gap(self, detections, frame_height) -> bool:
        """
        Определение наличия пропуска на основе детекций

        Args:
            detections: список детекций хлеба
            frame_height: высота кадра

        Returns:
            True если обнаружен значительный пропуск
        """
        if not detections:
            self.gap_history.append(True)
        else:
            # Анализируем распределение объектов по высоте
            y_positions = [d['center'][1] for d in detections]

            if len(y_positions) < 2:
                self.gap_history.append(False)
                return False

            # Сортируем по Y координате
            y_positions.sort()

            # Ищем большие пропуски между объектами
            gaps = []
            for i in range(1, len(y_positions)):
                gap = y_positions[i] - y_positions[i - 1]
                gaps.append(gap)

            # Оцениваем размер пропусков
            if gaps:
                max_gap = max(gaps)
                avg_object_height = frame_height * 0.05  # Примерная высота объекта

                # Пропуск считается значительным если больше N рядов
                gap_in_rows = max_gap / avg_object_height

                is_gap = gap_in_rows >= self.min_gap_rows
                self.gap_history.append(is_gap)

                if is_gap:
                    self.logger.info(f"Обнаружен пропуск: {gap_in_rows:.1f} рядов")

                return is_gap

        self.gap_history.append(False)
        return False

    def is_sustained_gap(self, min_duration_frames=10) -> bool:
        """Проверка на устойчивый пропуск"""
        if len(self.gap_history) < min_duration_frames:
            return False

        recent_gaps = list(self.gap_history)[-min_duration_frames:]
        return sum(recent_gaps) >= min_duration_frames * 0.7  # 70% кадров с пропуском


class SmartBatchDetector:
    """Умный детектор смены партий"""

    def __init__(self, oven_id: int):
        self.oven_id = oven_id
        self.gap_detector = GapDetector()
        self.bread_classifier = BreadTypeClassifier()

        # Состояние текущей партии
        self.current_batch_type = None
        self.current_batch_characteristics = None
        self.batch_start_time = None

        # История для принятия решений
        self.recent_detections = deque(maxlen=50)
        self.type_history = deque(maxlen=20)

        # Флаги состояния
        self.in_gap = False
        self.gap_start_time = None
        self.waiting_for_new_batch = False

        self.logger = logging.getLogger(f'SmartBatchDetector_Oven_{oven_id}')

    def process_frame(self, detections, frame, timestamp) -> Dict[str, Any]:
        """
        Обработка кадра для определения смены партий

        Returns:
            Dict с информацией о состоянии партии
        """
        frame_height = frame.shape[0]

        # 1. Детекция пропуска
        current_gap = self.gap_detector.detect_gap(detections, frame_height)

        # 2. Обработка состояний
        if current_gap and not self.in_gap:
            # Начало пропуска
            self.in_gap = True
            self.gap_start_time = timestamp
            self.waiting_for_new_batch = True
            self.logger.info("🔄 Начат пропуск - ожидание новой партии")

        elif not current_gap and self.in_gap:
            # Конец пропуска
            gap_duration = timestamp - self.gap_start_time if self.gap_start_time else 0
            self.in_gap = False
            self.logger.info(f"✅ Пропуск завершен (длительность: {gap_duration:.1f}с)")

        # 3. Анализ типа хлеба после пропуска
        if self.waiting_for_new_batch and not current_gap and detections:
            new_batch_type = self._analyze_new_batch(detections, frame)

            if new_batch_type != self.current_batch_type:
                # Обнаружена смена партии
                self._start_new_batch(new_batch_type, timestamp)

        # 4. Обновление истории
        self.recent_detections.append({
            'timestamp': timestamp,
            'detections': detections,
            'gap': current_gap
        })

        return {
            'batch_type': self.current_batch_type,
            'batch_name': self.bread_classifier.get_bread_name(self.current_batch_type or 'unknown'),
            'in_gap': self.in_gap,
            'waiting_for_new_batch': self.waiting_for_new_batch,
            'batch_start_time': self.batch_start_time,
            'gap_duration': timestamp - self.gap_start_time if self.gap_start_time else 0
        }

    def _analyze_new_batch(self, detections, frame) -> Optional[str]:
        """Анализ типа новой партии"""
        if not detections:
            return None

        # Анализируем несколько первых объектов
        sample_size = min(3, len(detections))
        characteristics = []

        for detection in detections[:sample_size]:
            if 'bbox' in detection:
                char = self.bread_classifier.extract_characteristics(frame, detection['bbox'])
                characteristics.append(char)

        if not characteristics:
            return None

        # Определяем тип по большинству
        types = []
        for char in characteristics:
            bread_type = self.bread_classifier.classify_bread_type(char)
            types.append(bread_type)

        # Наиболее частый тип
        if types:
            most_common = max(set(types), key=types.count)
            self.type_history.append(most_common)

            # Подтверждение типа на нескольких кадрах
            if len(self.type_history) >= 3:
                recent_types = list(self.type_history)[-3:]
                if recent_types.count(most_common) >= 2:
                    return most_common

        return None

    def _start_new_batch(self, batch_type: str, timestamp: float):
        """Начало новой партии"""
        old_type = self.current_batch_type
        old_name = self.bread_classifier.get_bread_name(old_type or 'unknown')
        new_name = self.bread_classifier.get_bread_name(batch_type)

        self.current_batch_type = batch_type
        self.batch_start_time = timestamp
        self.waiting_for_new_batch = False

        self.logger.info(f"🥖 СМЕНА ПАРТИИ: {old_name} → {new_name}")

        return {
            'event': 'batch_changed',
            'old_type': old_type,
            'new_type': batch_type,
            'old_name': old_name,
            'new_name': new_name,
            'timestamp': timestamp
        }

    def get_current_batch_info(self) -> Dict[str, Any]:
        """Получение информации о текущей партии"""
        duration = time.time() - self.batch_start_time if self.batch_start_time else 0

        return {
            'type': self.current_batch_type,
            'name': self.bread_classifier.get_bread_name(self.current_batch_type or 'unknown'),
            'start_time': self.batch_start_time,
            'duration': duration,
            'in_gap': self.in_gap,
            'waiting_for_new_batch': self.waiting_for_new_batch
        }

    def force_new_batch(self, batch_type: str = None):
        """Принудительное начало новой партии"""
        timestamp = time.time()

        if batch_type:
            self._start_new_batch(batch_type, timestamp)
        else:
            # Сброс состояния для автоопределения
            self.current_batch_type = None
            self.waiting_for_new_batch = True
            self.batch_start_time = timestamp

        self.logger.info(f"🔄 Принудительная смена партии: {batch_type or 'автоопределение'}")

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика работы детектора"""
        gap_ratio = sum(self.gap_detector.gap_history) / len(
            self.gap_detector.gap_history) if self.gap_detector.gap_history else 0

        return {
            'total_frames_processed': len(self.recent_detections),
            'gap_detection_ratio': gap_ratio,
            'current_batch_duration': time.time() - self.batch_start_time if self.batch_start_time else 0,
            'type_changes_detected': len(set(self.type_history)) if self.type_history else 0
        }