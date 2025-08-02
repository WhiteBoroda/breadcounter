# bread_tracker.py - Продвинутый трекинг объектов
from collections import defaultdict, deque
import numpy as np
import cv2
import math
import time


class BreadTracker:
    """Продвинутый трекер хлебобулочных изделий"""

    def __init__(self, oven_id, max_disappeared=30, max_distance=100):
        self.oven_id = oven_id
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Зоны подсчета
        self.counting_lines = []
        self.counted_tracks = set()

        # История для валидации
        self.detection_history = deque(maxlen=100)

    def add_counting_line(self, y_position, direction='down'):
        """Добавление линии подсчета"""
        self.counting_lines.append({
            'y': y_position,
            'direction': direction,
            'counted_ids': set(),
            'recent_crossings': deque(maxlen=50)  # История пересечений
        })

        print(f"📏 Добавлена линия подсчета на Y={y_position} для печи {self.oven_id}")

    def register(self, centroid, detection_data):
        """Регистрация нового объекта"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'positions': deque([centroid], maxlen=10),
            'counted': False,
            'first_seen': time.time(),
            'confidence_history': deque([detection_data.get('confidence', 0.5)], maxlen=10),
            'class_name': detection_data.get('class_name', 'unknown'),
            'bbox': detection_data.get('bbox', (0, 0, 0, 0)),
            'area': self._calculate_area(detection_data.get('bbox', (0, 0, 0, 0))),
            'velocity': (0, 0),  # Скорость движения
            'stable_frames': 0  # Количество стабильных кадров
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

        return self.next_id - 1

    def deregister(self, object_id):
        """Удаление объекта из трекера"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections):
        """Обновление трекера с новыми детекциями"""
        current_time = time.time()

        # Сохраняем в историю
        self.detection_history.append({
            'timestamp': current_time,
            'detections_count': len(detections),
            'detections': detections
        })

        if len(detections) == 0:
            # Увеличиваем счетчик исчезновения для всех объектов
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Извлекаем центроиды из детекций
        input_centroids = []
        detection_data = []

        for detection in detections:
            cx, cy = detection['center']
            input_centroids.append((cx, cy))
            detection_data.append(detection)

        if len(self.objects) == 0:
            # Регистрируем все детекции как новые объекты
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, detection_data[i])
        else:
            # Сопоставляем существующие объекты с новыми детекциями
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            # Вычисляем матрицу расстояний
            D = self._compute_distance_matrix(object_centroids, input_centroids)

            # Находим минимальные расстояния
            if D.size > 0:
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_row_indices = set()
                used_col_indices = set()

                for (row, col) in zip(rows, cols):
                    if row in used_row_indices or col in used_col_indices:
                        continue

                    if D[row, col] > self.max_distance:
                        continue

                    # Обновляем объект
                    object_id = object_ids[row]
                    old_centroid = self.objects[object_id]['centroid']
                    new_centroid = input_centroids[col]

                    # Обновляем данные объекта
                    self._update_object(object_id, new_centroid, detection_data[col], old_centroid)

                    # Проверяем пересечение линий подсчета
                    self._check_counting_lines(object_id)

                    self.disappeared[object_id] = 0

                    used_row_indices.add(row)
                    used_col_indices.add(col)

                # Обрабатываем непривязанные детекции и объекты
                unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
                unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)

                if D.shape[0] >= D.shape[1]:
                    # Больше объектов чем детекций - увеличиваем счетчик исчезновения
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
                else:
                    # Больше детекций чем объектов - регистрируем новые
                    for col in unused_cols:
                        self.register(input_centroids[col], detection_data[col])

        return self.objects

    def _update_object(self, object_id, new_centroid, detection_data, old_centroid):
        """Обновление данных объекта"""
        obj = self.objects[object_id]

        # Обновляем позицию
        obj['centroid'] = new_centroid
        obj['positions'].append(new_centroid)

        # Вычисляем скорость
        if len(obj['positions']) >= 2:
            prev_pos = obj['positions'][-2]
            obj['velocity'] = (
                new_centroid[0] - prev_pos[0],
                new_centroid[1] - prev_pos[1]
            )

        # Обновляем уверенность
        obj['confidence_history'].append(detection_data.get('confidence', 0.5))

        # Обновляем класс (если изменился)
        obj['class_name'] = detection_data.get('class_name', obj['class_name'])
        obj['bbox'] = detection_data.get('bbox', obj['bbox'])
        obj['area'] = self._calculate_area(obj['bbox'])

        # Проверяем стабильность движения
        if self._calculate_distance(old_centroid, new_centroid) < 10:
            obj['stable_frames'] += 1
        else:
            obj['stable_frames'] = 0

    def _compute_distance_matrix(self, object_centroids, input_centroids):
        """Вычисление матрицы расстояний между объектами и детекциями"""
        if not object_centroids or not input_centroids:
            return np.array([])

        D = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
            axis=2
        )
        return D

    def _calculate_distance(self, point1, point2):
        """Вычисление расстояния между двумя точками"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _calculate_area(self, bbox):
        """Вычисление площади bounding box"""
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            return (x2 - x1) * (y2 - y1)
        return 0

    def _check_counting_lines(self, object_id):
        """Проверка пересечения объектом линий подсчета"""
        obj = self.objects[object_id]
        positions = list(obj['positions'])

        if len(positions) < 2:
            return False

        prev_pos = positions[-2]
        curr_pos = positions[-1]

        crossed_lines = []

        for i, line in enumerate(self.counting_lines):
            line_y = line['y']

            # Проверяем пересечение линии
            if ((prev_pos[1] <= line_y <= curr_pos[1]) or
                    (curr_pos[1] <= line_y <= prev_pos[1])):

                if object_id not in line['counted_ids']:
                    # Дополнительные проверки для избежания ложных срабатываний
                    if self._validate_crossing(obj, line):
                        line['counted_ids'].add(object_id)
                        line['recent_crossings'].append({
                            'object_id': object_id,
                            'timestamp': time.time(),
                            'confidence': np.mean(obj['confidence_history']),
                            'class_name': obj['class_name']
                        })

                        obj['counted'] = True
                        crossed_lines.append(i)

                        print(f"✅ Объект {object_id} пересек линию {i} (печь {self.oven_id})")

        return len(crossed_lines) > 0

    def _validate_crossing(self, obj, line):
        """Валидация пересечения линии"""
        # Проверяем минимальную уверенность
        avg_confidence = np.mean(obj['confidence_history'])
        if avg_confidence < 0.3:
            return False

        # Проверяем минимальное время жизни объекта
        if time.time() - obj['first_seen'] < 0.5:
            return False

        # Проверяем размер объекта
        if obj['area'] < 1000:  # Слишком маленький объект
            return False

        # Проверяем направление движения (опционально)
        if line.get('direction') == 'down' and obj['velocity'][1] < 0:
            return False
        elif line.get('direction') == 'up' and obj['velocity'][1] > 0:
            return False

        return True

    def get_count_stats(self):
        """Получение статистики подсчета"""
        stats = {}
        total_counted = 0

        for i, line in enumerate(self.counting_lines):
            line_count = len(line['counted_ids'])
            stats[f'line_{i}'] = line_count
            total_counted += line_count

        # Статистика по классам
        class_stats = defaultdict(int)
        for line in self.counting_lines:
            for crossing in line['recent_crossings']:
                class_stats[crossing['class_name']] += 1

        stats['total'] = total_counted
        stats['by_class'] = dict(class_stats)
        stats['active_objects'] = len(self.objects)

        return stats

    def get_detection_quality_stats(self):
        """Статистика качества детекции"""
        if not self.detection_history:
            return {}

        recent_detections = list(self.detection_history)[-10:]  # Последние 10 кадров

        avg_detections = np.mean([d['detections_count'] for d in recent_detections])
        avg_confidence = 0

        confidence_values = []
        for detection_frame in recent_detections:
            for detection in detection_frame['detections']:
                confidence_values.append(detection.get('confidence', 0))

        if confidence_values:
            avg_confidence = np.mean(confidence_values)

        return {
            'avg_detections_per_frame': avg_detections,
            'avg_confidence': avg_confidence,
            'tracking_objects': len(self.objects),
            'stable_objects': len([obj for obj in self.objects.values() if obj['stable_frames'] > 5])
        }

    def reset_counts(self):
        """Сброс счетчиков (для новой партии)"""
        for line in self.counting_lines:
            line['counted_ids'].clear()
            line['recent_crossings'].clear()

        # Сбрасываем флаги подсчета у объектов
        for obj in self.objects.values():
            obj['counted'] = False

        print(f"🔄 Счетчики сброшены для печи {self.oven_id}")

    def setup_counting_zones(self, frame_height, frame_width):
        """Настройка зон подсчета на основе размера кадра"""
        # Очищаем существующие зоны
        self.counting_lines.clear()

        # Создаем 3 линии через конвейер для верификации
        line_positions = [
            int(frame_height * 0.3),  # 30% от высоты
            int(frame_height * 0.5),  # 50% от высоты
            int(frame_height * 0.7)  # 70% от высоты
        ]

        for pos in line_positions:
            self.add_counting_line(pos, 'down')

        print(f"🎯 Настроены зоны подсчета для печи {self.oven_id}: {line_positions}")

    def draw_tracking_info(self, frame):
        """Отрисовка информации трекинга на кадре"""
        # Рисуем линии подсчета
        for i, line in enumerate(self.counting_lines):
            y = line['y']
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
            cv2.line(frame, (0, y), (frame.shape[1], y), color, 2)

            # Подписываем линию
            cv2.putText(frame, f"Line {i + 1}: {len(line['counted_ids'])}",
                        (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Рисуем трекинг объектов
        for object_id, obj in self.objects.items():
            center = (int(obj['centroid'][0]), int(obj['centroid'][1]))

            # Цвет в зависимости от класса
            if obj['class_name'] == 'bread':
                color = (0, 255, 0)
            elif obj['class_name'] in ['circle', 'square', 'triangle']:
                color = (255, 255, 0)
            else:
                color = (128, 128, 128)

            # Рисуем центр
            cv2.circle(frame, center, 8, color, -1)

            # ID объекта
            cv2.putText(frame, str(object_id),
                        (center[0] + 15, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Траектория
            if len(obj['positions']) > 1:
                points = np.array([(int(p[0]), int(p[1])) for p in obj['positions']],
                                  np.int32)
                cv2.polylines(frame, [points], False, color, 2)

        # Общая статистика
        stats = self.get_count_stats()
        y_offset = 30
        for key, value in stats.items():
            if key != 'by_class':
                cv2.putText(frame, f"{key}: {value}",
                            (frame.shape[1] - 200, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

        return frame