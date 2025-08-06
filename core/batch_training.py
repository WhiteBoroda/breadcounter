# core/batch_training.py
"""Система пакетного обучения с детекцией аномалий и брака"""

import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os


class BatchTrainingManager:
    """Менеджер пакетного обучения с детекцией аномалий"""

    def __init__(self):
        self.current_batch = None
        self.batch_template = None
        self.similarity_threshold = 0.8  # Порог схожести с шаблоном
        self.defect_categories = {
            'merged': {'name': 'Слипшиеся', 'color': [255, 0, 255]},
            'deformed': {'name': 'Деформированные', 'color': [255, 165, 0]},
            'undercooked': {'name': 'Недопеченные', 'color': [139, 69, 19]},
            'overcooked': {'name': 'Перепеченные', 'color': [64, 64, 64]},
            'size_anomaly': {'name': 'Неправильный размер', 'color': [255, 20, 147]},
            'foreign_object': {'name': 'Посторонний объект', 'color': [128, 0, 128]}
        }
        self.auto_training_active = False
        self.anomaly_queue = []

    def create_batch(self, batch_info: Dict) -> str:
        """Создание новой партии для обучения"""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_batch = {
            'id': batch_id,
            'product_info': batch_info,
            'created': datetime.now().isoformat(),
            'template_objects': [],
            'processed_objects': 0,
            'good_objects': 0,
            'defect_objects': 0,
            'anomaly_objects': 0,
            'defect_breakdown': {cat: 0 for cat in self.defect_categories},
            'status': 'created'
        }

        self._save_batch()
        return batch_id

    def set_batch_template(self, template_objects: List[Dict]) -> bool:
        """Установка эталонных объектов для партии"""
        if not self.current_batch:
            return False

        # Анализируем эталонные объекты для создания шаблона
        self.batch_template = self._create_template_from_objects(template_objects)
        self.current_batch['template_objects'] = template_objects
        self.current_batch['status'] = 'template_ready'

        self._save_batch()
        return True

    def start_auto_training(self) -> bool:
        """Запуск автоматического обучения"""
        if not self.batch_template:
            return False

        self.auto_training_active = True
        self.current_batch['status'] = 'auto_training'
        self.current_batch['auto_training_started'] = datetime.now().isoformat()

        self._save_batch()
        return True

    def process_detected_objects(self, objects: List[Dict], frame_data: Dict) -> Dict:
        """Обработка обнаруженных объектов с автоклассификацией"""
        if not self.auto_training_active or not self.batch_template:
            return {'status': 'error', 'message': 'Автообучение не активно'}

        results = {
            'processed': 0,
            'good': 0,
            'defects': 0,
            'anomalies': 0,
            'stop_required': False,
            'anomaly_objects': [],
            'defect_objects': [],
            'classifications': []
        }

        for obj in objects:
            classification = self._classify_object(obj, frame_data)
            results['classifications'].append(classification)
            results['processed'] += 1

            if classification['type'] == 'good':
                results['good'] += 1
                self._add_to_training_data(obj, classification, frame_data)

            elif classification['type'] == 'defect':
                results['defects'] += 1
                results['defect_objects'].append({
                    'object': obj,
                    'classification': classification,
                    'frame': frame_data
                })
                self._add_defect_to_training_data(obj, classification, frame_data)

            elif classification['type'] == 'anomaly':
                results['anomalies'] += 1
                results['anomaly_objects'].append({
                    'object': obj,
                    'classification': classification,
                    'frame': frame_data
                })
                # Аномалия требует остановки для ручной классификации
                results['stop_required'] = True
                self.anomaly_queue.append({
                    'object': obj,
                    'classification': classification,
                    'frame': frame_data,
                    'timestamp': datetime.now().isoformat()
                })

        # Обновляем статистику партии
        self._update_batch_stats(results)

        if results['stop_required']:
            self.auto_training_active = False
            self.current_batch['status'] = 'stopped_anomaly'
            self._save_batch()

        return results

    def resolve_anomaly(self, anomaly_id: int, resolution: Dict) -> bool:
        """Разрешение аномалии оператором"""
        if anomaly_id >= len(self.anomaly_queue):
            return False

        anomaly = self.anomaly_queue[anomaly_id]

        if resolution['action'] == 'add_to_good':
            # Добавить к хорошим объектам и обновить шаблон
            self._add_to_training_data(anomaly['object'], {
                'type': 'good',
                'category': self.current_batch['product_info']['category'],
                'confidence': 1.0,
                'operator_verified': True
            }, anomaly['frame'])
            self._update_template_with_object(anomaly['object'])

        elif resolution['action'] == 'mark_as_defect':
            # Отметить как брак
            defect_classification = {
                'type': 'defect',
                'defect_category': resolution['defect_category'],
                'confidence': 1.0,
                'operator_verified': True
            }
            self._add_defect_to_training_data(anomaly['object'], defect_classification, anomaly['frame'])

        elif resolution['action'] == 'ignore':
            # Игнорировать объект
            pass

        # Удаляем из очереди
        self.anomaly_queue.pop(anomaly_id)

        # Если очередь пуста, можно возобновить автообучение
        if not self.anomaly_queue and resolution.get('resume_training', False):
            self.auto_training_active = True
            self.current_batch['status'] = 'auto_training'
            self._save_batch()

        return True

    def _classify_object(self, obj: Dict, frame_data: Dict) -> Dict:
        """Классификация объекта относительно шаблона партии"""
        # Извлекаем признаки объекта
        features = self._extract_object_features(obj, frame_data)

        # Сравниваем с шаблоном
        similarity = self._calculate_similarity(features, self.batch_template)

        if similarity >= self.similarity_threshold:
            # Объект похож на шаблон
            return {
                'type': 'good',
                'category': self.current_batch['product_info']['category'],
                'similarity': similarity,
                'confidence': similarity
            }

        # Проверяем на известные типы брака
        defect_type = self._detect_defect_type(features)
        if defect_type:
            return {
                'type': 'defect',
                'defect_category': defect_type,
                'similarity': similarity,
                'confidence': 0.8
            }

        # Неизвестный объект - аномалия
        return {
            'type': 'anomaly',
            'similarity': similarity,
            'confidence': 0.5,
            'reason': 'Unknown object type'
        }

    def _extract_object_features(self, obj: Dict, frame_data: Dict) -> Dict:
        """Извлечение признаков объекта для сравнения"""
        bbox = obj['bbox']

        # Базовые геометрические признаки
        features = {
            'width': bbox['width'],
            'height': bbox['height'],
            'area': bbox['width'] * bbox['height'],
            'aspect_ratio': bbox['width'] / bbox['height'],
            'center': obj.get('center', {'x': bbox['x'] + bbox['width'] // 2, 'y': bbox['y'] + bbox['height'] // 2})
        }

        # Если есть изображение кадра, добавляем цветовые признаки
        if 'image' in frame_data:
            roi = self._extract_roi(frame_data['image'], bbox)
            color_features = self._analyze_colors(roi)
            features.update(color_features)

        return features

    def _calculate_similarity(self, features: Dict, template: Dict) -> float:
        """Расчет схожести объекта с шаблоном"""
        if not template:
            return 0.0

        # Нормализованные признаки для сравнения
        size_similarity = self._compare_sizes(features, template)
        shape_similarity = self._compare_shapes(features, template)
        color_similarity = self._compare_colors(features, template)

        # Взвешенная схожесть
        total_similarity = (
                size_similarity * 0.3 +
                shape_similarity * 0.4 +
                color_similarity * 0.3
        )

        return total_similarity

    def _detect_defect_type(self, features: Dict) -> Optional[str]:
        """Определение типа брака по признакам"""
        # Слипшиеся хлеба (большая площадь, неправильное соотношение сторон)
        if features['area'] > self.batch_template['avg_area'] * 1.8:
            if features['aspect_ratio'] > 2.0 or features['aspect_ratio'] < 0.5:
                return 'merged'

        # Деформированные (неправильное соотношение сторон)
        template_ratio = self.batch_template['avg_aspect_ratio']
        if abs(features['aspect_ratio'] - template_ratio) > template_ratio * 0.4:
            return 'deformed'

        # Неправильный размер
        if features['area'] < self.batch_template['min_area'] or features['area'] > self.batch_template['max_area']:
            return 'size_anomaly'

        # Цветовые аномалии (если есть цветовая информация)
        if 'avg_hue' in features and 'avg_hue' in self.batch_template:
            hue_diff = abs(features['avg_hue'] - self.batch_template['avg_hue'])
            if hue_diff > 30:  # Разница в оттенке больше 30 градусов
                if features['avg_brightness'] < 0.3:
                    return 'overcooked'
                elif features['avg_brightness'] > 0.8:
                    return 'undercooked'

        return None

    def _create_template_from_objects(self, objects: List[Dict]) -> Dict:
        """Создание шаблона из эталонных объектов"""
        if not objects:
            return {}

        # Собираем статистику по всем объектам
        areas = [obj['bbox']['width'] * obj['bbox']['height'] for obj in objects]
        aspect_ratios = [obj['bbox']['width'] / obj['bbox']['height'] for obj in objects]

        template = {
            'count': len(objects),
            'avg_area': np.mean(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'std_area': np.std(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'min_aspect_ratio': np.min(aspect_ratios),
            'max_aspect_ratio': np.max(aspect_ratios),
            'std_aspect_ratio': np.std(aspect_ratios)
        }

        return template

    def _compare_sizes(self, features: Dict, template: Dict) -> float:
        """Сравнение размеров объекта с шаблоном"""
        area_diff = abs(features['area'] - template['avg_area'])
        max_acceptable_diff = template['std_area'] * 2

        if area_diff <= max_acceptable_diff:
            return 1.0 - (area_diff / max_acceptable_diff)
        return 0.0

    def _compare_shapes(self, features: Dict, template: Dict) -> float:
        """Сравнение формы объекта с шаблоном"""
        ratio_diff = abs(features['aspect_ratio'] - template['avg_aspect_ratio'])
        max_acceptable_diff = template['std_aspect_ratio'] * 2

        if ratio_diff <= max_acceptable_diff:
            return 1.0 - (ratio_diff / max_acceptable_diff)
        return 0.0

    def _compare_colors(self, features: Dict, template: Dict) -> float:
        """Сравнение цветов объекта с шаблоном"""
        if 'avg_hue' not in features or 'avg_hue' not in template:
            return 1.0  # Если цветовой информации нет, считаем похожими

        hue_diff = abs(features['avg_hue'] - template['avg_hue'])
        brightness_diff = abs(features['avg_brightness'] - template['avg_brightness'])

        # Нормализуем различия
        hue_similarity = max(0, 1.0 - hue_diff / 180.0)  # Максимальная разность 180 градусов
        brightness_similarity = max(0, 1.0 - brightness_diff)

        return (hue_similarity + brightness_similarity) / 2

    def _extract_roi(self, image: np.ndarray, bbox: Dict) -> np.ndarray:
        """Извлечение области интереса из изображения"""
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        return image[y:y + h, x:x + w]

    def _analyze_colors(self, roi: np.ndarray) -> Dict:
        """Анализ цветовых характеристик области"""
        if roi.size == 0:
            return {}

        # Конвертируем в HSV для анализа
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Средние значения
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_brightness = np.mean(hsv[:, :, 2]) / 255.0

        return {
            'avg_hue': avg_hue,
            'avg_saturation': avg_saturation,
            'avg_brightness': avg_brightness
        }

    def _add_to_training_data(self, obj: Dict, classification: Dict, frame_data: Dict):
        """Добавление объекта в данные для обучения"""
        training_record = {
            'object_id': obj['id'],
            'bbox': obj['bbox'],
            'classification': classification,
            'frame_index': frame_data.get('frame_index'),
            'timestamp': datetime.now().isoformat(),
            'batch_id': self.current_batch['id']
        }

        # Сохраняем в файл обучающих данных
        self._save_training_record(training_record)

    def _add_defect_to_training_data(self, obj: Dict, classification: Dict, frame_data: Dict):
        """Добавление брака в данные для обучения"""
        defect_record = {
            'object_id': obj['id'],
            'bbox': obj['bbox'],
            'defect_type': classification['defect_category'],
            'classification': classification,
            'frame_index': frame_data.get('frame_index'),
            'timestamp': datetime.now().isoformat(),
            'batch_id': self.current_batch['id']
        }

        # Сохраняем в файл данных о браке
        self._save_defect_record(defect_record)

    def _update_batch_stats(self, results: Dict):
        """Обновление статистики партии"""
        self.current_batch['processed_objects'] += results['processed']
        self.current_batch['good_objects'] += results['good']
        self.current_batch['defect_objects'] += results['defects']
        self.current_batch['anomaly_objects'] += results['anomalies']

        # Обновляем разбивку по типам брака
        for classification in results['classifications']:
            if classification['type'] == 'defect':
                defect_cat = classification['defect_category']
                if defect_cat in self.current_batch['defect_breakdown']:
                    self.current_batch['defect_breakdown'][defect_cat] += 1

    def _update_template_with_object(self, obj: Dict):
        """Обновление шаблона с учетом нового объекта"""
        # Здесь можно реализовать адаптивное обновление шаблона
        pass

    def _save_batch(self):
        """Сохранение информации о партии"""
        if not self.current_batch:
            return

        os.makedirs('training_data/batches', exist_ok=True)
        filename = f"training_data/batches/{self.current_batch['id']}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_batch, f, ensure_ascii=False, indent=2)

    def _save_training_record(self, record: Dict):
        """Сохранение записи обучающих данных"""
        os.makedirs('training_data/good_objects', exist_ok=True)
        filename = f"training_data/good_objects/{record['batch_id']}_good.jsonl"

        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    def _save_defect_record(self, record: Dict):
        """Сохранение записи о браке"""
        os.makedirs('training_data/defects', exist_ok=True)
        filename = f"training_data/defects/{record['batch_id']}_defects.jsonl"

        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    def get_batch_statistics(self) -> Dict:
        """Получение статистики текущей партии"""
        if not self.current_batch:
            return {}

        total = self.current_batch['processed_objects']
        if total == 0:
            return self.current_batch

        stats = self.current_batch.copy()
        stats.update({
            'good_percentage': (self.current_batch['good_objects'] / total) * 100,
            'defect_percentage': (self.current_batch['defect_objects'] / total) * 100,
            'anomaly_percentage': (self.current_batch['anomaly_objects'] / total) * 100
        })

        return stats

    def get_pending_anomalies(self) -> List[Dict]:
        """Получение списка аномалий, ожидающих разрешения"""
        return [
            {
                'id': i,
                'object': anomaly['object'],
                'classification': anomaly['classification'],
                'timestamp': anomaly['timestamp']
            }
            for i, anomaly in enumerate(self.anomaly_queue)
        ]