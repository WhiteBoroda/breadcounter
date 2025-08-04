# production_zone_training.py - Производственная система зонной разметки
"""
Производственная система зонной разметки с интеграцией учетной системы.
Работа через SKU коды, производственные задания и реальные партии.
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime, date
from pathlib import Path
import base64
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Date, Decimal, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Product(Base):
    """Продукты из учетной системы (справочник SKU)"""
    __tablename__ = 'products'

    sku_code = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    weight = Column(Decimal(5, 3))  # Вес в кг
    category = Column(String(100))
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

    # Визуальные характеристики для обучения
    shape_type = Column(String(50))  # round, oval, rectangular
    color_profile = Column(String(100))  # light, dark, mixed
    surface_type = Column(String(50))  # smooth, textured, scored
    typical_size_min = Column(Integer)  # мин размер в пикселях
    typical_size_max = Column(Integer)  # макс размер в пикселях

    def __repr__(self):
        return f"<Product(sku='{self.sku_code}', name='{self.name}', weight={self.weight})>"


class ProductionOrder(Base):
    """Производственные задания"""
    __tablename__ = 'production_orders'

    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, nullable=False)
    shift_date = Column(Date, nullable=False)
    shift_number = Column(Integer, nullable=False)  # 1, 2, 3
    oven_id = Column(Integer, nullable=False)
    sku_code = Column(String(50), ForeignKey('products.sku_code'), nullable=False)
    target_quantity = Column(Integer, nullable=False)
    priority = Column(Integer, default=1)
    status = Column(String(50), default='planned')  # planned, active, completed, cancelled
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Связи
    product = relationship("Product")
    batches = relationship("ProductionBatch", back_populates="production_order")

    def __repr__(self):
        return f"<ProductionOrder(order='{self.order_number}', sku='{self.sku_code}', qty={self.target_quantity})>"


class ProductionBatch(Base):
    """Производственные партии (активные выпечки)"""
    __tablename__ = 'production_batches'

    id = Column(Integer, primary_key=True)
    production_order_id = Column(Integer, ForeignKey('production_orders.id'), nullable=False)
    batch_number = Column(String(50), nullable=False)
    oven_id = Column(Integer, nullable=False)

    # Плановые показатели
    expected_quantity = Column(Integer, nullable=False)
    expected_weight_per_unit = Column(Decimal(5, 3))

    # Фактические показатели
    actual_count = Column(Integer, default=0)
    quality_grade = Column(String(20), default='A')  # A, B, C, defective

    # Временные метки
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)

    # Статус
    status = Column(String(50), default='active')  # active, paused, completed, cancelled

    # Метаданные для обучения
    training_video_path = Column(String(500))
    zones_config_path = Column(String(500))
    model_training_completed = Column(Boolean, default=False)

    # Связи
    production_order = relationship("ProductionOrder", back_populates="batches")
    training_sessions = relationship("TrainingSession", back_populates="batch")

    def __repr__(self):
        return f"<ProductionBatch(batch='{self.batch_number}', oven={self.oven_id}, count={self.actual_count})>"


class TrainingSession(Base):
    """Сессии обучения для конкретных партий"""
    __tablename__ = 'training_sessions'

    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey('production_batches.id'), nullable=False)
    session_name = Column(String(200), nullable=False)

    # Параметры обучения
    video_filename = Column(String(500))
    frames_annotated = Column(Integer, default=0)
    objects_detected = Column(Integer, default=0)
    training_started = Column(DateTime)
    training_completed = Column(DateTime)

    # Зоны разметки
    zones_config = Column(Text)  # JSON с зонами
    detection_params = Column(Text)  # JSON с параметрами детекции

    # Результаты
    model_accuracy = Column(Decimal(5, 4))
    validation_score = Column(Decimal(5, 4))

    status = Column(String(50), default='created')  # created, annotating, training, completed, failed

    # Связи
    batch = relationship("ProductionBatch", back_populates="training_sessions")

    def __repr__(self):
        return f"<TrainingSession(session='{self.session_name}', status='{self.status}')>"


class ProductionZoneManager:
    """Производственный менеджер зон с интеграцией учетной системы"""

    def __init__(self, db_url="sqlite:///production_system.db"):
        # База данных
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db = Session()

        # Текущая сессия обучения
        self.current_session = None
        self.current_batch = None
        self.current_product = None

        # Зоны разметки
        self.zones = {
            'counting_zone': None,
            'entry_zone': None,
            'exit_zone': None,
            'exclude_zones': []
        }

        # Параметры детекции (настраиваются под продукт)
        self.detection_params = {
            'min_area': 2000,
            'max_area': 25000,
            'hsv_lower': [10, 20, 20],
            'hsv_upper': [30, 255, 200],
            'morphology_kernel_size': 3,
            'blur_kernel_size': 3
        }

        # Инициализация тестовых данных
        self._init_test_data()

    def _init_test_data(self):
        """Инициализация тестовых данных для демонстрации"""
        # Проверяем есть ли уже данные
        if self.db.query(Product).count() > 0:
            return

        # Добавляем тестовые продукты
        test_products = [
            Product(
                sku_code="BRD001",
                name="Олександрівський формовий 0.7кг",
                weight=0.7,
                category="Хлеб формовый",
                shape_type="rectangular",
                color_profile="light",
                surface_type="smooth",
                typical_size_min=3000,
                typical_size_max=8000
            ),
            Product(
                sku_code="BRD002",
                name="Дарницький житній 0.5кг",
                weight=0.5,
                category="Хлеб житний",
                shape_type="oval",
                color_profile="dark",
                surface_type="textured",
                typical_size_min=2500,
                typical_size_max=6000
            ),
            Product(
                sku_code="BTN001",
                name="Український батон 0.4кг",
                weight=0.4,
                category="Батони",
                shape_type="oval",
                color_profile="light",
                surface_type="scored",
                typical_size_min=2000,
                typical_size_max=5000
            ),
            Product(
                sku_code="BRD003",
                name="Бородинський з кмином 0.8кг",
                weight=0.8,
                category="Хлеб спеціальний",
                shape_type="round",
                color_profile="dark",
                surface_type="textured",
                typical_size_min=4000,
                typical_size_max=9000
            )
        ]

        for product in test_products:
            self.db.add(product)

        # Добавляем тестовые производственные задания
        today = date.today()
        test_orders = [
            ProductionOrder(
                order_number="ПЗ-001-240801",
                shift_date=today,
                shift_number=1,
                oven_id=1,
                sku_code="BRD001",
                target_quantity=800,
                status="active"
            ),
            ProductionOrder(
                order_number="ПЗ-002-240801",
                shift_date=today,
                shift_number=1,
                oven_id=2,
                sku_code="BRD002",
                target_quantity=1200,
                status="planned"
            ),
            ProductionOrder(
                order_number="ПЗ-003-240801",
                shift_date=today,
                shift_number=2,
                oven_id=1,
                sku_code="BTN001",
                target_quantity=600,
                status="planned"
            )
        ]

        for order in test_orders:
            self.db.add(order)

        self.db.commit()
        print("✅ Тестовые данные инициализированы")

    def get_available_products(self):
        """Получение списка доступных продуктов"""
        products = self.db.query(Product).filter(Product.is_active == True).all()
        return [
            {
                'sku_code': p.sku_code,
                'name': p.name,
                'weight': float(p.weight) if p.weight else 0,
                'category': p.category,
                'shape_type': p.shape_type,
                'color_profile': p.color_profile
            }
            for p in products
        ]

    def get_active_production_orders(self, oven_id=None):
        """Получение активных производственных заданий"""
        query = self.db.query(ProductionOrder).filter(
            ProductionOrder.status.in_(['planned', 'active'])
        )

        if oven_id:
            query = query.filter(ProductionOrder.oven_id == oven_id)

        orders = query.join(Product).all()

        return [
            {
                'id': order.id,
                'order_number': order.order_number,
                'sku_code': order.sku_code,
                'product_name': order.product.name,
                'weight': float(order.product.weight) if order.product.weight else 0,
                'target_quantity': order.target_quantity,
                'oven_id': order.oven_id,
                'shift_date': order.shift_date.isoformat(),
                'shift_number': order.shift_number,
                'status': order.status,
                'priority': order.priority
            }
            for order in orders
        ]

    def create_training_session(self, production_order_id, session_name, video_filename):
        """Создание новой сессии обучения"""
        try:
            # Получаем производственное задание
            production_order = self.db.query(ProductionOrder).get(production_order_id)
            if not production_order:
                return {'success': False, 'error': 'Производственное задание не найдено'}

            # Создаем или получаем партию
            batch = self.db.query(ProductionBatch).filter(
                ProductionBatch.production_order_id == production_order_id,
                ProductionBatch.status == 'active'
            ).first()

            if not batch:
                # Создаем новую партию
                batch = ProductionBatch(
                    production_order_id=production_order_id,
                    batch_number=f"БТЧ-{production_order.order_number}-{int(time.time())}",
                    oven_id=production_order.oven_id,
                    expected_quantity=production_order.target_quantity,
                    expected_weight_per_unit=production_order.product.weight
                )
                self.db.add(batch)
                self.db.flush()

            # Создаем сессию обучения
            training_session = TrainingSession(
                batch_id=batch.id,
                session_name=session_name,
                video_filename=video_filename,
                status='created'
            )

            self.db.add(training_session)
            self.db.commit()

            # Устанавливаем как текущую сессию
            self.current_session = training_session
            self.current_batch = batch
            self.current_product = production_order.product

            # Настраиваем параметры детекции под продукт
            self._configure_detection_for_product(self.current_product)

            return {
                'success': True,
                'session_id': training_session.id,
                'batch_info': {
                    'batch_number': batch.batch_number,
                    'sku_code': production_order.sku_code,
                    'product_name': production_order.product.name,
                    'expected_quantity': batch.expected_quantity,
                    'weight': float(production_order.product.weight) if production_order.product.weight else 0
                }
            }

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def _configure_detection_for_product(self, product):
        """Настройка параметров детекции под конкретный продукт"""
        if not product:
            return

        # Настраиваем размеры объектов
        if product.typical_size_min:
            self.detection_params['min_area'] = product.typical_size_min
        if product.typical_size_max:
            self.detection_params['max_area'] = product.typical_size_max

        # Настраиваем цветовой профиль
        if product.color_profile == 'dark':
            self.detection_params['hsv_lower'] = [5, 20, 10]
            self.detection_params['hsv_upper'] = [25, 255, 150]
        elif product.color_profile == 'light':
            self.detection_params['hsv_lower'] = [15, 20, 40]
            self.detection_params['hsv_upper'] = [35, 255, 220]

        print(f"🎯 Настроены параметры детекции для {product.name}")

    def load_training_session(self, session_id):
        """Загрузка существующей сессии обучения"""
        try:
            session = self.db.query(TrainingSession).get(session_id)
            if not session:
                return {'success': False, 'error': 'Сессия не найдена'}

            self.current_session = session
            self.current_batch = session.batch
            self.current_product = session.batch.production_order.product

            # Загружаем зоны если есть
            if session.zones_config:
                self.zones = json.loads(session.zones_config)

            # Загружаем параметры детекции если есть
            if session.detection_params:
                self.detection_params.update(json.loads(session.detection_params))
            else:
                self._configure_detection_for_product(self.current_product)

            return {
                'success': True,
                'session': {
                    'id': session.id,
                    'name': session.session_name,
                    'status': session.status,
                    'frames_annotated': session.frames_annotated,
                    'objects_detected': session.objects_detected
                },
                'batch_info': {
                    'batch_number': self.current_batch.batch_number,
                    'sku_code': self.current_product.sku_code,
                    'product_name': self.current_product.name,
                    'expected_quantity': self.current_batch.expected_quantity
                },
                'zones': self.zones
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def save_zones_config(self):
        """Сохранение конфигурации зон"""
        if not self.current_session:
            return {'success': False, 'error': 'Нет активной сессии'}

        try:
            self.current_session.zones_config = json.dumps(self.zones)
            self.current_session.detection_params = json.dumps(self.detection_params)
            self.db.commit()

            return {'success': True}

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def detect_objects_in_zones(self, frame):
        """Детекция объектов с учетом зон и характеристик продукта"""
        if not self.current_product:
            return []

        detections = self._detect_product_objects(frame)

        # Фильтруем детекции по зонам
        filtered_detections = []
        for detection in detections:
            center_x, center_y = detection['center']

            # Пропускаем объекты в зонах исключения
            if self._point_in_exclude_zones(center_x, center_y):
                continue

            # Помечаем принадлежность к зонам
            detection['in_counting_zone'] = self._point_in_zone(center_x, center_y, 'counting_zone')
            detection['in_entry_zone'] = self._point_in_zone(center_x, center_y, 'entry_zone')
            detection['in_exit_zone'] = self._point_in_zone(center_x, center_y, 'exit_zone')

            # Добавляем информацию о продукте
            detection['sku_code'] = self.current_product.sku_code
            detection['product_name'] = self.current_product.name
            detection['expected_weight'] = float(self.current_product.weight) if self.current_product.weight else 0

            filtered_detections.append(detection)

        return filtered_detections

    def _detect_product_objects(self, frame):
        """Детекция объектов с учетом характеристик продукта"""
        detections = []

        # HSV детекция с настройками под продукт
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.detection_params['hsv_lower'])
        upper = np.array(self.detection_params['hsv_upper'])
        mask = cv2.inRange(hsv, lower, upper)

        # Морфология с адаптивными параметрами
        kernel_size = self.detection_params.get('morphology_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Размытие
        blur_size = self.detection_params.get('blur_kernel_size', 3)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        # Контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Фильтрация по размеру для конкретного продукта
            if area < self.detection_params['min_area'] or area > self.detection_params['max_area']:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2

            # Дополнительная валидация по форме продукта
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0

            # Проверка соответствия форме продукта
            if not self._validate_shape_for_product(aspect_ratio, extent, area):
                continue

            detection = {
                'id': i,
                'bbox': [x, y, x + w, y + h],
                'center': [center_x, center_y],
                'area': area,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'confidence': self._calculate_confidence(area, aspect_ratio, extent)
            }
            detections.append(detection)

        return detections

    def _validate_shape_for_product(self, aspect_ratio, extent, area):
        """Валидация формы объекта для конкретного продукта"""
        if not self.current_product:
            return True

        shape_type = self.current_product.shape_type

        if shape_type == 'round':
            # Круглые изделия: соотношение сторон близко к 1, высокая плотность
            return 0.7 <= aspect_ratio <= 1.4 and extent >= 0.6
        elif shape_type == 'oval':
            # Овальные изделия: умеренное соотношение сторон
            return 0.6 <= aspect_ratio <= 2.0 and extent >= 0.5
        elif shape_type == 'rectangular':
            # Прямоугольные изделия: может быть вытянутой формы
            return 0.4 <= aspect_ratio <= 3.0 and extent >= 0.4
        else:
            # Универсальная валидация
            return 0.3 <= aspect_ratio <= 4.0 and extent >= 0.3

    def _calculate_confidence(self, area, aspect_ratio, extent):
        """Расчет уверенности детекции на основе характеристик продукта"""
        base_confidence = 0.5

        # Бонус за соответствие ожидаемому размеру
        if self.current_product and self.current_product.typical_size_min and self.current_product.typical_size_max:
            ideal_size = (self.current_product.typical_size_min + self.current_product.typical_size_max) / 2
            size_deviation = abs(area - ideal_size) / ideal_size
            size_bonus = max(0, 0.3 - size_deviation)
        else:
            size_bonus = 0.1

        # Бонус за хорошую форму
        shape_bonus = min(0.2, extent * 0.3)

        return min(0.95, base_confidence + size_bonus + shape_bonus)

    def generate_production_annotations(self, frame, frame_index, detections):
        """Генерация аннотаций в производственном формате"""
        if not self.current_session or not self.current_product:
            return []

        annotations = []

        # Фильтруем только объекты в зоне подсчета
        valid_detections = [d for d in detections if d.get('in_counting_zone', False)]

        for detection in valid_detections:
            x1, y1, x2, y2 = detection['bbox']

            annotation = {
                'bbox': [x1, y1, x2, y2],
                'center': detection['center'],
                'area': detection['area'],
                'confidence': detection['confidence'],

                # Производственная информация
                'sku_code': self.current_product.sku_code,
                'product_name': self.current_product.name,
                'expected_weight': float(self.current_product.weight) if self.current_product.weight else 0,
                'batch_number': self.current_batch.batch_number,
                'production_order': self.current_batch.production_order.order_number,

                # Метаданные
                'session_id': self.current_session.id,
                'frame_index': frame_index,
                'zone_validated': True,
                'detection_params': self.detection_params.copy()
            }
            annotations.append(annotation)

        return annotations

    def create_production_training_dataset(self, video_cap, frames_count=200):
        """Создание производственного датасета"""
        if not video_cap or not self.current_session:
            return {'success': False, 'error': 'Нет активной сессии или видео'}

        if not self.zones.get('counting_zone'):
            return {'success': False, 'error': 'Не определена зона подсчета'}

        try:
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // frames_count)

            generated_annotations = []
            processed_frames = 0
            total_objects = 0

            # Создаем папку для датасета
            dataset_name = f"production_dataset_{self.current_product.sku_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_path = f"training_data/production/{dataset_name}"
            os.makedirs(f"{dataset_path}/images", exist_ok=True)
            os.makedirs(f"{dataset_path}/annotations", exist_ok=True)

            for frame_idx in range(0, total_frames, step):
                if processed_frames >= frames_count:
                    break

                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video_cap.read()

                if not ret:
                    continue

                # Детекция с учетом зон
                detections = self.detect_objects_in_zones(frame)

                # Генерируем аннотации
                annotations = self.generate_production_annotations(frame, frame_idx, detections)

                if len(annotations) > 0:
                    # Сохраняем кадр
                    img_filename = f"frame_{frame_idx:06d}.jpg"
                    img_path = f"{dataset_path}/images/{img_filename}"
                    cv2.imwrite(img_path, frame)

                    # Сохраняем аннотацию
                    ann_filename = f"frame_{frame_idx:06d}.json"
                    ann_path = f"{dataset_path}/annotations/{ann_filename}"

                    annotation_data = {
                        'image_filename': img_filename,
                        'frame_index': frame_idx,
                        'timestamp': frame_idx / video_cap.get(cv2.CAV_PROP_FPS),
                        'annotations': annotations,
                        'production_info': {
                            'sku_code': self.current_product.sku_code,
                            'product_name': self.current_product.name,
                            'batch_number': self.current_batch.batch_number,
                            'session_id': self.current_session.id
                        }
                    }

                    with open(ann_path, 'w', encoding='utf-8') as f:
                        json.dump(annotation_data, f, ensure_ascii=False, indent=2)

                    generated_annotations.append(annotation_data)
                    processed_frames += 1
                    total_objects += len(annotations)

            # Обновляем статистику сессии
            self.current_session.frames_annotated = processed_frames
            self.current_session.objects_detected = total_objects
            self.current_session.status = 'annotated'

            # Сохраняем метаданные датасета
            metadata = {
                'created': datetime.now().isoformat(),
                'production_info': {
                    'sku_code': self.current_product.sku_code,
                    'product_name': self.current_product.name,
                    'batch_number': self.current_batch.batch_number,
                    'production_order': self.current_batch.production_order.order_number,
                    'expected_quantity': self.current_batch.expected_quantity
                },
                'training_info': {
                    'session_id': self.current_session.id,
                    'session_name': self.current_session.session_name,
                    'frames_generated': processed_frames,
                    'total_objects': total_objects,
                    'zones_config': self.zones,
                    'detection_params': self.detection_params
                },
                'dataset_structure': {
                    'images_path': f"{dataset_path}/images",
                    'annotations_path': f"{dataset_path}/annotations",
                    'format': 'production_json'
                }
            }

            with open(f"{dataset_path}/production_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.db.commit()

            return {
                'success': True,
                'dataset_path': dataset_path,
                'generated_frames': processed_frames,
                'total_objects': total_objects,
                'sku_code': self.current_product.sku_code,
                'product_name': self.current_product.name
            }

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def visualize_production_frame(self, frame, detections=None):
        """Визуализация кадра с производственной информацией"""
        annotated_frame = frame.copy()

        # Цвета зон
        zone_colors = {
            'counting_zone': (0, 255, 0),  # Зеленый
            'entry_zone': (255, 0, 0),  # Синий
            'exit_zone': (0, 0, 255),  # Красный
            'exclude_zones': (128, 128, 128)  # Серый
        }

        # Рисуем зоны
        for zone_name, color in zone_colors.items():
            if zone_name == 'exclude_zones':
                for zone in self.zones.get(zone_name, []):
                    if zone:
                        self._draw_zone(annotated_frame, zone, color)
            else:
                zone = self.zones.get(zone_name)
                if zone:
                    self._draw_zone(annotated_frame, zone, color)

        # Рисуем детекции
        if detections:
            for detection in detections:
                self._draw_production_detection(annotated_frame, detection)

        # Производственная информация
        if self.current_product and self.current_batch:
            info_lines = [
                f"SKU: {self.current_product.sku_code}",
                f"Продукт: {self.current_product.name}",
                f"Партия: {self.current_batch.batch_number}",
                f"План: {self.current_batch.expected_quantity} шт"
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(annotated_frame, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_frame

    def _draw_production_detection(self, frame, detection):
        """Отрисовка детекции с производственной информацией"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        center_x, center_y = detection['center']

        # Цвет в зависимости от зоны
        if detection.get('in_counting_zone'):
            color = (0, 255, 0)  # Зеленый - в зоне подсчета
            thickness = 3
        elif detection.get('in_entry_zone'):
            color = (255, 0, 0)  # Синий - в зоне входа
            thickness = 2
        elif detection.get('in_exit_zone'):
            color = (0, 0, 255)  # Красный - в зоне выхода
            thickness = 2
        else:
            color = (255, 255, 255)  # Белый - вне зон
            thickness = 1

        # Рамка и центр
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)

        # Информация об объекте
        confidence = detection.get('confidence', 0)
        area = detection.get('area', 0)
        weight = detection.get('expected_weight', 0)

        label = f"ID:{detection['id']} ({confidence:.2f}) {weight}кг"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Вспомогательные методы (аналогичны предыдущей версии)
    def _point_in_zone(self, x, y, zone_name):
        zone = self.zones.get(zone_name)
        if not zone or len(zone) < 3:
            return False
        points = np.array(zone, np.int32)
        return cv2.pointPolygonTest(points, (x, y), False) >= 0

    def _point_in_exclude_zones(self, x, y):
        for zone in self.zones.get('exclude_zones', []):
            if zone and len(zone) >= 3:
                points = np.array(zone, np.int32)
                if cv2.pointPolygonTest(points, (x, y), False) >= 0:
                    return True
        return False

    def _draw_zone(self, frame, zone, color):
        if not zone or len(zone) < 3:
            return
        points = np.array(zone, np.int32)

        # Полупрозрачная заливка
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Контур
        cv2.polylines(frame, [points], True, color, 2)

    def get_training_sessions(self, limit=20):
        """Получение списка сессий обучения"""
        sessions = self.db.query(TrainingSession).order_by(TrainingSession.id.desc()).limit(limit).all()

        return [
            {
                'id': session.id,
                'session_name': session.session_name,
                'status': session.status,
                'batch_number': session.batch.batch_number,
                'product_name': session.batch.production_order.product.name,
                'sku_code': session.batch.production_order.sku_code,
                'frames_annotated': session.frames_annotated,
                'objects_detected': session.objects_detected,
                'created': session.batch.start_time.isoformat() if session.batch.start_time else None
            }
            for session in sessions
        ]

    def close(self):
        """Закрытие соединения с БД"""
        if self.db:
            self.db.close()


# Flask API маршруты для производственной системы
def add_production_zone_routes(app, zone_manager):
    """Добавление маршрутов производственной зонной разметки"""

    @app.route('/api/production/products')
    def get_products():
        """Получение списка продуктов"""
        products = zone_manager.get_available_products()
        return jsonify({'success': True, 'products': products})

    @app.route('/api/production/orders')
    def get_production_orders():
        """Получение производственных заданий"""
        oven_id = request.args.get('oven_id', type=int)
        orders = zone_manager.get_active_production_orders(oven_id)
        return jsonify({'success': True, 'orders': orders})

    @app.route('/api/production/create_session', methods=['POST'])
    def create_training_session():
        """Создание сессии обучения"""
        data = request.get_json()
        production_order_id = data.get('production_order_id')
        session_name = data.get('session_name')
        video_filename = data.get('video_filename')

        if not all([production_order_id, session_name, video_filename]):
            return jsonify({'success': False, 'error': 'Не все параметры указаны'})

        result = zone_manager.create_training_session(production_order_id, session_name, video_filename)
        return jsonify(result)

    @app.route('/api/production/load_session', methods=['POST'])
    def load_training_session():
        """Загрузка сессии обучения"""
        data = request.get_json()
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'success': False, 'error': 'ID сессии не указан'})

        result = zone_manager.load_training_session(session_id)
        return jsonify(result)

    @app.route('/api/production/sessions')
    def get_training_sessions():
        """Получение списка сессий обучения"""
        sessions = zone_manager.get_training_sessions()
        return jsonify({'success': True, 'sessions': sessions})

    @app.route('/api/production/save_zones', methods=['POST'])
    def save_zones():
        """Сохранение зон"""
        data = request.get_json()
        zones = data.get('zones', {})

        zone_manager.zones = zones
        result = zone_manager.save_zones_config()
        return jsonify(result)

    @app.route('/api/production/generate_dataset', methods=['POST'])
    def generate_dataset():
        """Генерация производственного датасета"""
        data = request.get_json()
        frames_count = data.get('frames_count', 200)
        video_cap = getattr(app, 'current_video_cap', None)

        if not video_cap:
            return jsonify({'success': False, 'error': 'Видео не загружено'})

        result = zone_manager.create_production_training_dataset(video_cap, frames_count)
        return jsonify(result)

    @app.route('/api/production/visualize_frame', methods=['POST'])
    def visualize_frame():
        """Визуализация кадра с производственной информацией"""
        data = request.get_json()
        frame_index = data.get('frame_index', 0)
        video_cap = getattr(app, 'current_video_cap', None)

        if not video_cap:
            return jsonify({'success': False, 'error': 'Видео не загружено'})

        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_cap.read()

        if not ret:
            return jsonify({'success': False, 'error': 'Не удалось прочитать кадр'})

        # Детекция и визуализация
        detections = zone_manager.detect_objects_in_zones(frame)
        annotated_frame = zone_manager.visualize_production_frame(frame, detections)

        # Конвертация в base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_base64,
            'detections': detections,
            'stats': {
                'total_objects': len(detections),
                'in_counting_zone': len([d for d in detections if d.get('in_counting_zone')]),
                'in_entry_zone': len([d for d in detections if d.get('in_entry_zone')]),
                'in_exit_zone': len([d for d in detections if d.get('in_exit_zone')])
            },
            'production_info': {
                'sku_code': zone_manager.current_product.sku_code if zone_manager.current_product else None,
                'product_name': zone_manager.current_product.name if zone_manager.current_product else None,
                'batch_number': zone_manager.current_batch.batch_number if zone_manager.current_batch else None
            }
        })


if __name__ == '__main__':
    # Пример использования
    zone_manager = ProductionZoneManager()

    print("🏭 Производственная система зонной разметки инициализирована")
    print("📊 Доступные продукты:")

    products = zone_manager.get_available_products()
    for product in products:
        print(f"   • {product['sku_code']}: {product['name']} ({product['weight']}кг)")

    print("\n📋 Активные производственные задания:")
    orders = zone_manager.get_active_production_orders()
    for order in orders:
        print(
            f"   • {order['order_number']}: {order['product_name']} - {order['target_quantity']} шт (Печь {order['oven_id']})")

    print("\n🔧 Для интеграции с Flask app:")
    print("   from production_zone_training import add_production_zone_routes, ProductionZoneManager")
    print("   zone_manager = ProductionZoneManager()")
    print("   add_production_zone_routes(app, zone_manager)")

    zone_manager.close()