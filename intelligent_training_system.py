# intelligent_training_system.py - Интеллектуальная система обучения
"""
Умная система обучения с автоматическими предложениями зон и интерактивным обучением.
Алгоритм:
1. Загрузка видео → автоматическое предложение зон
2. Корректировка зон пользователем
3. Система находит объекты → пользователь выбирает и описывает продукт
4. Интерактивное обучение с вопросами системы
5. Создание новых продуктов "на лету"
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import base64
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import DECIMAL

Base = declarative_base()


class TrainingProduct(Base):
    """Продукты созданные в процессе обучения"""
    __tablename__ = 'training_products'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    sku_code = Column(String(50), unique=True, nullable=False)
    weight = Column(DECIMAL(5, 3))
    description = Column(Text)

    # Визуальные характеристики (обучаемые)
    avg_area = Column(Integer)
    avg_width = Column(Integer)
    avg_height = Column(Integer)
    color_profile = Column(String(100))  # HSV диапазон
    shape_features = Column(Text)  # JSON с характеристиками формы

    # Статистика обучения
    samples_count = Column(Integer, default=0)
    confidence_threshold = Column(DECIMAL(3, 2), default=0.7)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Связи
    samples = relationship("TrainingSample", back_populates="product")


class TrainingSample(Base):
    """Образцы для обучения"""
    __tablename__ = 'training_samples'

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('training_products.id'), nullable=False)

    # Данные образца
    frame_index = Column(Integer, nullable=False)
    bbox = Column(String(100))  # JSON: [x1, y1, x2, y2]
    center = Column(String(50))  # JSON: [cx, cy]
    area = Column(Integer)

    # Характеристики
    hsv_values = Column(Text)  # JSON с HSV характеристиками
    shape_features = Column(Text)  # JSON с характеристиками формы

    # Метаданные
    is_validated = Column(Boolean, default=False)
    user_confirmed = Column(Boolean, default=True)
    quality_score = Column(DECIMAL(3, 2))  # Оценка качества образца
    created_at = Column(DateTime, default=func.now())

    # Связи
    product = relationship("TrainingProduct", back_populates="samples")


class InteractiveTrainingSession(Base):
    """Сессия интерактивного обучения"""
    __tablename__ = 'interactive_training_sessions'

    id = Column(Integer, primary_key=True)
    session_name = Column(String(200), nullable=False)
    video_filename = Column(String(500))

    # Зоны (автоматически предложенные + скорректированные)
    suggested_zones = Column(Text)  # JSON с предложенными зонами
    final_zones = Column(Text)  # JSON с финальными зонами

    # Прогресс обучения
    total_frames = Column(Integer)
    processed_frames = Column(Integer, default=0)
    identified_objects = Column(Integer, default=0)
    user_interactions = Column(Integer, default=0)  # Количество вопросов к пользователю

    # Статус
    status = Column(String(50), default='zone_setup')  # zone_setup, learning, completed
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)


class IntelligentTrainingManager:
    """Менеджер интеллектуального обучения"""

    def __init__(self, db_url="sqlite:///intelligent_training.db"):
        # База данных
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db = Session()

        # Текущая сессия
        self.current_session = None
        self.current_video_cap = None
        self.total_frames = 0

        # Зоны
        self.suggested_zones = {}
        self.final_zones = {}

        # Обучение
        self.learning_queue = []  # Очередь объектов для классификации пользователем
        self.current_products = {}  # Кэш продуктов для быстрого доступа

    def create_training_session(self, session_name, video_path):
        """Создание новой сессии обучения"""
        try:
            # Загружаем видео
            self.current_video_cap = cv2.VideoCapture(video_path)
            if not self.current_video_cap.isOpened():
                return {'success': False, 'error': 'Не удалось открыть видео'}

            self.total_frames = int(self.current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Создаем сессию
            session = InteractiveTrainingSession(
                session_name=session_name,
                video_filename=os.path.basename(video_path),
                total_frames=self.total_frames
            )

            self.db.add(session)
            self.db.commit()
            self.current_session = session

            # Автоматически предлагаем зоны
            suggested_zones = self._auto_suggest_zones()
            session.suggested_zones = json.dumps(suggested_zones)
            self.db.commit()

            return {
                'success': True,
                'session_id': session.id,
                'total_frames': self.total_frames,
                'suggested_zones': suggested_zones
            }

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def _auto_suggest_zones(self):
        """Автоматическое предложение зон детекции"""
        if not self.current_video_cap:
            return {}

        # Анализируем несколько кадров для предложения зон
        sample_frames = []
        frame_indices = [0, self.total_frames // 4, self.total_frames // 2, 3 * self.total_frames // 4]

        for frame_idx in frame_indices:
            self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.current_video_cap.read()
            if ret:
                sample_frames.append(frame)

        if not sample_frames:
            return {}

        # Анализируем движение и находим активные зоны
        h, w = sample_frames[0].shape[:2]

        # Простая эвристика для предложения зон
        suggested_zones = {
            'counting_zone': [
                [w // 6, h // 3],  # Левый верх
                [5 * w // 6, h // 3],  # Правый верх
                [5 * w // 6, 2 * h // 3],  # Правый низ
                [w // 6, 2 * h // 3]  # Левый низ
            ],
            'entry_zone': [
                [0, h // 4],
                [w // 4, h // 4],
                [w // 4, 3 * h // 4],
                [0, 3 * h // 4]
            ],
            'exit_zone': [
                [3 * w // 4, h // 4],
                [w, h // 4],
                [w, 3 * h // 4],
                [3 * w // 4, 3 * h // 4]
            ]
        }

        return suggested_zones

    def finalize_zones(self, zones):
        """Финализация зон после корректировки пользователем"""
        if not self.current_session:
            return {'success': False, 'error': 'Нет активной сессии'}

        try:
            self.final_zones = zones
            self.current_session.final_zones = json.dumps(zones)
            self.current_session.status = 'learning'
            self.db.commit()

            # Начинаем поиск объектов для обучения
            self._start_object_discovery()

            return {'success': True}

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def _start_object_discovery(self):
        """Начало поиска и анализа объектов"""
        if not self.current_video_cap or not self.final_zones.get('counting_zone'):
            return

        # Сканируем видео с интервалом и ищем объекты
        step = max(1, self.total_frames // 100)  # Анализируем каждый 100-й кадр

        for frame_idx in range(0, self.total_frames, step):
            self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.current_video_cap.read()

            if not ret:
                continue

            # Находим объекты на кадре
            objects = self._detect_objects_on_frame(frame)

            # Фильтруем по зоне подсчета
            zone_objects = self._filter_objects_by_zone(objects, self.final_zones['counting_zone'])

            # Добавляем в очередь для классификации
            for obj in zone_objects:
                obj['frame_index'] = frame_idx
                obj['frame'] = frame  # Сохраняем кадр для показа пользователю
                self.learning_queue.append(obj)

        print(f"Найдено {len(self.learning_queue)} объектов для обучения")

    def _detect_objects_on_frame(self, frame):
        """Детекция объектов на кадре"""
        objects = []

        # HSV детекция (расширенный диапазон для хлебных оттенков)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Несколько масок для разных типов хлеба
        masks = []

        # Светлый хлеб
        masks.append(cv2.inRange(hsv, np.array([10, 20, 40]), np.array([35, 255, 220])))

        # Темный хлеб
        masks.append(cv2.inRange(hsv, np.array([5, 20, 10]), np.array([25, 255, 150])))

        # Батоны (более светлые)
        masks.append(cv2.inRange(hsv, np.array([15, 10, 60]), np.array([40, 200, 255])))

        # Объединяем маски
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Находим контуры
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 1500 < area < 30000:  # Расширенный диапазон размеров
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2

                # Анализируем характеристики объекта
                roi = frame[y:y + h, x:x + w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                obj = {
                    'bbox': [x, y, x + w, y + h],
                    'center': [center_x, center_y],
                    'area': area,
                    'width': w,
                    'height': h,
                    'aspect_ratio': w / h if h > 0 else 0,
                    'roi': roi,
                    'hsv_stats': {
                        'mean_hue': float(np.mean(hsv_roi[:, :, 0])),
                        'mean_saturation': float(np.mean(hsv_roi[:, :, 1])),
                        'mean_value': float(np.mean(hsv_roi[:, :, 2]))
                    }
                }
                objects.append(obj)

        return objects

    def _filter_objects_by_zone(self, objects, zone):
        """Фильтрация объектов по зоне"""
        if not zone or len(zone) < 3:
            return objects

        zone_objects = []
        points = np.array(zone, np.int32)

        for obj in objects:
            center_x, center_y = obj['center']
            if cv2.pointPolygonTest(points, (center_x, center_y), False) >= 0:
                zone_objects.append(obj)

        return zone_objects

    def get_next_object_for_classification(self):
        """Получение следующего объекта для классификации пользователем"""
        if not self.learning_queue:
            return None

        obj = self.learning_queue.pop(0)

        # Конвертируем ROI в base64 для показа пользователю
        roi = obj['roi']
        _, buffer = cv2.imencode('.jpg', roi)
        roi_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'object_data': obj,
            'roi_image': roi_base64,
            'remaining_objects': len(self.learning_queue),
            'characteristics': {
                'area': obj['area'],
                'dimensions': f"{obj['width']}x{obj['height']}",
                'aspect_ratio': round(obj['aspect_ratio'], 2),
                'color_info': obj['hsv_stats']
            }
        }

    def classify_object(self, object_data, classification):
        """Классификация объекта пользователем"""
        try:
            if classification['type'] == 'new_product':
                # Создаем новый продукт
                product = self._create_new_product(classification['product_info'])
                self._add_training_sample(product.id, object_data, classification)

            elif classification['type'] == 'existing_product':
                # Добавляем к существующему продукту
                product_id = classification['product_id']
                self._add_training_sample(product_id, object_data, classification)

            elif classification['type'] == 'defective':
                # Создаем образец брака
                self._handle_defective_product(object_data, classification)

            elif classification['type'] == 'not_product':
                # Игнорируем (лоток с водой, посторонний предмет)
                pass

            # Обновляем статистику сессии
            if self.current_session:
                self.current_session.identified_objects += 1
                self.current_session.user_interactions += 1
                self.db.commit()

            return {'success': True}

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def _create_new_product(self, product_info):
        """Создание нового продукта"""
        product = TrainingProduct(
            name=product_info['name'],
            sku_code=product_info['sku_code'],
            weight=float(product_info.get('weight', 0)),
            description=product_info.get('description', '')
        )

        self.db.add(product)
        self.db.flush()  # Получаем ID

        self.current_products[product.id] = product
        return product

    def _add_training_sample(self, product_id, object_data, classification):
        """Добавление образца для обучения"""
        sample = TrainingSample(
            product_id=product_id,
            frame_index=object_data['frame_index'],
            bbox=json.dumps(object_data['bbox']),
            center=json.dumps(object_data['center']),
            area=object_data['area'],
            hsv_values=json.dumps(object_data['hsv_stats']),
            shape_features=json.dumps({
                'width': object_data['width'],
                'height': object_data['height'],
                'aspect_ratio': object_data['aspect_ratio']
            }),
            quality_score=classification.get('quality_score', 1.0)
        )

        self.db.add(sample)

        # Обновляем статистику продукта
        product = self.db.query(TrainingProduct).get(product_id)
        if product:
            product.samples_count += 1
            # Обновляем средние значения
            self._update_product_statistics(product)

    def _update_product_statistics(self, product):
        """Обновление статистики продукта на основе образцов"""
        samples = self.db.query(TrainingSample).filter(
            TrainingSample.product_id == product.id
        ).all()

        if samples:
            areas = [s.area for s in samples]
            product.avg_area = int(np.mean(areas))

            # Обновляем HSV профиль
            hsv_values = []
            for sample in samples:
                hsv_data = json.loads(sample.hsv_values)
                hsv_values.append(hsv_data)

            if hsv_values:
                avg_hue = np.mean([h['mean_hue'] for h in hsv_values])
                avg_sat = np.mean([h['mean_saturation'] for h in hsv_values])
                avg_val = np.mean([h['mean_value'] for h in hsv_values])

                product.color_profile = json.dumps({
                    'hue_range': [avg_hue - 10, avg_hue + 10],
                    'saturation_range': [max(0, avg_sat - 50), min(255, avg_sat + 50)],
                    'value_range': [max(0, avg_val - 50), min(255, avg_val + 50)]
                })

    def get_training_progress(self):
        """Получение прогресса обучения"""
        if not self.current_session:
            return {}

        products = self.db.query(TrainingProduct).all()

        return {
            'session_id': self.current_session.id,
            'session_name': self.current_session.session_name,
            'status': self.current_session.status,
            'total_frames': self.current_session.total_frames,
            'processed_frames': self.current_session.processed_frames,
            'identified_objects': self.current_session.identified_objects,
            'user_interactions': self.current_session.user_interactions,
            'remaining_objects': len(self.learning_queue),
            'products_learned': len(products),
            'products': [
                {
                    'id': p.id,
                    'name': p.name,
                    'sku_code': p.sku_code,
                    'weight': float(p.weight) if p.weight else 0,
                    'samples_count': p.samples_count
                }
                for p in products
            ]
        }

    def generate_final_dataset(self):
        """Генерация финального датасета после обучения"""
        if not self.current_session:
            return {'success': False, 'error': 'Нет активной сессии'}

        try:
            dataset_name = f"intelligent_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_path = f"training_data/intelligent/{dataset_name}"

            os.makedirs(f"{dataset_path}/images", exist_ok=True)
            os.makedirs(f"{dataset_path}/annotations", exist_ok=True)

            # Получаем все образцы
            samples = self.db.query(TrainingSample).join(TrainingProduct).all()

            generated_samples = 0

            for sample in samples:
                # Получаем кадр
                self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
                ret, frame = self.current_video_cap.read()

                if not ret:
                    continue

                # Сохраняем изображение
                img_filename = f"sample_{sample.id:06d}.jpg"
                img_path = f"{dataset_path}/images/{img_filename}"
                cv2.imwrite(img_path, frame)

                # Создаем аннотацию
                bbox = json.loads(sample.bbox)
                annotation = {
                    'image_filename': img_filename,
                    'sample_id': sample.id,
                    'product_info': {
                        'id': sample.product.id,
                        'name': sample.product.name,
                        'sku_code': sample.product.sku_code,
                        'weight': float(sample.product.weight) if sample.product.weight else 0
                    },
                    'bbox': bbox,
                    'center': json.loads(sample.center),
                    'area': sample.area,
                    'hsv_values': json.loads(sample.hsv_values),
                    'shape_features': json.loads(sample.shape_features),
                    'quality_score': float(sample.quality_score) if sample.quality_score else 1.0
                }

                ann_filename = f"sample_{sample.id:06d}.json"
                ann_path = f"{dataset_path}/annotations/{ann_filename}"

                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, ensure_ascii=False, indent=2)

                generated_samples += 1

            # Сохраняем метаданные датасета
            products = self.db.query(TrainingProduct).all()
            metadata = {
                'created': datetime.now().isoformat(),
                'session_info': {
                    'id': self.current_session.id,
                    'name': self.current_session.session_name,
                    'video': self.current_session.video_filename
                },
                'zones': json.loads(self.current_session.final_zones),
                'products': [
                    {
                        'id': p.id,
                        'name': p.name,
                        'sku_code': p.sku_code,
                        'weight': float(p.weight) if p.weight else 0,
                        'samples_count': p.samples_count,
                        'color_profile': json.loads(p.color_profile) if p.color_profile else None
                    }
                    for p in products
                ],
                'statistics': {
                    'total_samples': generated_samples,
                    'total_products': len(products),
                    'user_interactions': self.current_session.user_interactions
                }
            }

            with open(f"{dataset_path}/intelligent_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # Завершаем сессию
            self.current_session.status = 'completed'
            self.current_session.completed_at = func.now()
            self.db.commit()

            return {
                'success': True,
                'dataset_path': dataset_path,
                'generated_samples': generated_samples,
                'products_count': len(products)
            }

        except Exception as e:
            self.db.rollback()
            return {'success': False, 'error': str(e)}

    def close(self):
        """Закрытие ресурсов"""
        if self.current_video_cap:
            self.current_video_cap.release()
        if self.db:
            self.db.close()


# Flask API для интеллектуальной системы обучения
def add_intelligent_training_routes(app, training_manager):
    """Добавление маршрутов интеллектуальной системы обучения"""

    @app.route('/api/intelligent/create_session', methods=['POST'])
    def create_intelligent_session():
        data = request.get_json()
        session_name = data.get('session_name')

        # Получаем путь к текущему видео из app или запроса
        video_path = getattr(app, 'current_video_path', None)
        if not video_path:
            # Пытаемся получить из атрибута current_video приложения
            current_video = getattr(app, 'current_video', None)
            if current_video:
                video_path = current_video

        print(f"DEBUG: session_name={session_name}, video_path={video_path}")

        if not session_name:
            return jsonify({'success': False, 'error': 'Введите название сессии'})

        if not video_path:
            return jsonify({'success': False, 'error': 'Сначала загрузите видео'})

        try:
            result = training_manager.create_training_session(session_name, video_path)
            print(f"DEBUG: create_training_session result={result}")
            return jsonify(result)
        except Exception as e:
            print(f"DEBUG: Exception in create_training_session: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/intelligent/finalize_zones', methods=['POST'])
    def finalize_zones():
        data = request.get_json()
        zones = data.get('zones', {})

        result = training_manager.finalize_zones(zones)
        return jsonify(result)

    @app.route('/api/intelligent/get_next_object')
    def get_next_object():
        obj = training_manager.get_next_object_for_classification()
        if obj:
            return jsonify({'success': True, 'object': obj})
        else:
            return jsonify({'success': False, 'message': 'Нет объектов для классификации'})

    @app.route('/api/intelligent/classify_object', methods=['POST'])
    def classify_object():
        data = request.get_json()
        object_data = data.get('object_data')
        classification = data.get('classification')

        result = training_manager.classify_object(object_data, classification)
        return jsonify(result)

    @app.route('/api/intelligent/get_progress')
    def get_training_progress():
        progress = training_manager.get_training_progress()
        return jsonify({'success': True, 'progress': progress})

    @app.route('/api/intelligent/generate_dataset', methods=['POST'])
    def generate_intelligent_dataset():
        result = training_manager.generate_final_dataset()
        return jsonify(result)


# JavaScript для интеллектуального интерфейса
INTELLIGENT_TRAINING_JS = '''
// intelligent_training.js - Интеллектуальная система обучения

class IntelligentTrainingInterface {
    constructor() {
        this.currentSession = null;
        this.suggestedZones = {};
        this.finalZones = {};
        this.currentObject = null;
        this.isDrawing = false;
        this.currentTool = null;
        this.currentZone = [];
        this.trainingStep = 'upload'; // upload, zones, learning, completed
    }

    initializeInterface() {
        const controlPanel = document.querySelector('.control-panel') || document.getElementById('control-panel');

        if (controlPanel) {
            const interfaceHTML = `
                <div class="intelligent-training" style="padding: 15px; border: 2px solid #27ae60; border-radius: 10px;">
                    <h3>🤖 Интеллектуальное обучение</h3>

                    <!-- Шаг 1: Начало сессии -->
                    <div id="step-start" class="training-step">
                        <h4>Шаг 1: Создание сессии обучения</h4>
                        <input type="text" id="sessionName" placeholder="Название сессии" 
                               style="width: 100%; padding: 8px; margin: 5px 0; background: #34495e; border: 1px solid #27ae60; color: white; border-radius: 4px;">
                        <button class="btn success" id="startIntelligentSession" style="width: 100%; margin: 10px 0;">
                            🚀 Начать обучение
                        </button>
                    </div>

                    <!-- Шаг 2: Настройка зон -->
                    <div id="step-zones" class="training-step" style="display: none;">
                        <h4>Шаг 2: Настройка зон детекции</h4>
                        <p style="color: #bdc3c7; font-size: 12px;">Система предложила зоны. Скорректируйте их или оставьте как есть.</p>

                        <div class="zone-tools" style="display: grid; gap: 8px; margin: 10px 0;">
                            <button class="zone-btn" data-zone="counting_zone" style="background: rgba(39, 174, 96, 0.8); border: none; padding: 8px; color: white; border-radius: 4px;">
                                🟢 Зона подсчета
                            </button>
                            <button class="zone-btn" data-zone="entry_zone" style="background: rgba(52, 152, 219, 0.8); border: none; padding: 8px; color: white; border-radius: 4px;">
                                🔵 Зона входа
                            </button>
                            <button class="zone-btn" data-zone="exit_zone" style="background: rgba(231, 76, 60, 0.8); border: none; padding: 8px; color: white; border-radius: 4px;">
                                🔴 Зона выхода
                            </button>
                        </div>

                        <div id="zoneInstructions" style="background: #f39c12; color: #2c3e50; padding: 10px; border-radius: 5px; margin: 10px 0; display: none;">
                            <strong>Корректировка зоны:</strong><br>
                            • Кликайте по углам для изменения<br>
                            • Двойной клик - завершить<br>
                            • ESC - отменить
                        </div>

                        <button class="btn success" id="finalizeZones" style="width: 100%; margin: 10px 0;">
                            ✅ Зоны готовы, начать обучение
                        </button>
                    </div>

                    <!-- Шаг 3: Интерактивное обучение -->
                    <div id="step-learning" class="training-step" style="display: none;">
                        <h4>Шаг 3: Обучение системы</h4>
                        <div id="learningProgress" style="background: rgba(52, 73, 94, 0.8); padding: 10px; border-radius: 5px; margin: 10px 0;">
                            <p>Инициализация...</p>
                        </div>

                        <!-- Объект для классификации -->
                        <div id="objectClassification" style="display: none;">
                            <h5>❓ Что это за объект?</h5>
                            <div id="objectImage" style="text-align: center; margin: 10px 0;">
                                <!-- Изображение объекта -->
                            </div>
                            <div id="objectInfo" style="background: rgba(52, 73, 94, 0.8); padding: 8px; border-radius: 4px; margin: 10px 0;">
                                <!-- Характеристики объекта -->
                            </div>

                            <!-- Варианты классификации -->
                            <div class="classification-options" style="display: grid; gap: 8px; margin: 10px 0;">
                                <button class="btn success" id="newProduct">➕ Новый продукт</button>
                                <button class="btn" id="existingProduct">📦 Существующий продукт</button>
                                <button class="btn danger" id="defectiveProduct">❌ Брак</button>
                                <button class="btn secondary" id="notProduct">🚫 Не продукт</button>
                            </div>
                        </div>

                        <!-- Форма нового продукта -->
                        <div id="newProductForm" style="display: none; background: rgba(52, 73, 94, 0.8); padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <h5>📝 Описание нового продукта:</h5>
                            <input type="text" id="productName" placeholder="Название (например: Олександрівський формовий)" 
                                   style="width: 100%; padding: 8px; margin: 5px 0; background: #2c3e50; border: 1px solid #27ae60; color: white; border-radius: 4px;">
                            <input type="text" id="productSKU" placeholder="SKU код (например: BRD001)" 
                                   style="width: 100%; padding: 8px; margin: 5px 0; background: #2c3e50; border: 1px solid #27ae60; color: white; border-radius: 4px;">
                            <input type="number" id="productWeight" placeholder="Вес в кг (например: 0.7)" step="0.1" 
                                   style="width: 100%; padding: 8px; margin: 5px 0; background: #2c3e50; border: 1px solid #27ae60; color: white; border-radius: 4px;">
                            <textarea id="productDescription" placeholder="Описание (опционально)" 
                                      style="width: 100%; padding: 8px; margin: 5px 0; background: #2c3e50; border: 1px solid #27ae60; color: white; border-radius: 4px; resize: vertical; height: 60px;"></textarea>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                                <button class="btn success" id="saveNewProduct">💾 Сохранить</button>
                                <button class="btn secondary" id="cancelNewProduct">❌ Отмена</button>
                            </div>
                        </div>

                        <!-- Список существующих продуктов -->
                        <div id="existingProductsList" style="display: none;">
                            <h5>📦 Выберите существующий продукт:</h5>
                            <div id="productsList" style="max-height: 200px; overflow-y: auto;">
                                <!-- Список продуктов -->
                            </div>
                        </div>
                    </div>

                    <!-- Шаг 4: Завершение -->
                    <div id="step-completed" class="training-step" style="display: none;">
                        <h4>🎉 Обучение завершено!</h4>
                        <div id="completionStats" style="background: rgba(39, 174, 96, 0.8); padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <!-- Статистика обучения -->
                        </div>
                        <button class="btn success" id="generateFinalDataset" style="width: 100%; margin: 10px 0;">
                            🚀 Создать финальный датасет
                        </button>
                    </div>
                </div>
            `;

            controlPanel.innerHTML = interfaceHTML;
            this.bindEvents();
        }
    }

    bindEvents() {
        // Создание сессии
        document.getElementById('startIntelligentSession')?.addEventListener('click', () => this.startSession());

        // Зонные инструменты
        document.querySelectorAll('.zone-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.selectZoneTool(e.target.dataset.zone));
        });

        document.getElementById('finalizeZones')?.addEventListener('click', () => this.finalizeZones());

        // Классификация объектов
        document.getElementById('newProduct')?.addEventListener('click', () => this.showNewProductForm());
        document.getElementById('existingProduct')?.addEventListener('click', () => this.showExistingProducts());
        document.getElementById('defectiveProduct')?.addEventListener('click', () => this.classifyAsDefective());
        document.getElementById('notProduct')?.addEventListener('click', () => this.classifyAsNotProduct());

        // Форма нового продукта
        document.getElementById('saveNewProduct')?.addEventListener('click', () => this.saveNewProduct());
        document.getElementById('cancelNewProduct')?.addEventListener('click', () => this.hideNewProductForm());

        document.getElementById('generateFinalDataset')?.addEventListener('click', () => this.generateFinalDataset());

        // Canvas events
        const canvas = document.getElementById('drawingCanvas') || document.querySelector('canvas');
        if (canvas) {
            canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
            canvas.addEventListener('dblclick', (e) => this.finishZone(e));
        }

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.cancelDrawing();
        });
    }

    async startSession() {
        const sessionName = document.getElementById('sessionName')?.value;
        if (!sessionName) {
            this.showStatus('Введите название сессии', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/intelligent/create_session', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_name: sessionName})
            });

            const data = await response.json();
            if (data.success) {
                this.currentSession = data.session_id;
                this.suggestedZones = data.suggested_zones;
                this.finalZones = {...this.suggestedZones};

                this.showStep('zones');
                this.drawSuggestedZones();
                this.showStatus('Сессия создана. Скорректируйте зоны или начните обучение.', 'success');
            } else {
                this.showStatus('Ошибка создания сессии: ' + data.error, 'error');
            }
        } catch (error) {
            this.showStatus('Ошибка: ' + error.message, 'error');
        }
    }

    showStep(step) {
        // Скрываем все шаги
        document.querySelectorAll('.training-step').forEach(el => el.style.display = 'none');

        // Показываем нужный шаг
        const stepElement = document.getElementById(`step-${step}`);
        if (stepElement) {
            stepElement.style.display = 'block';
            this.trainingStep = step;
        }
    }

    drawSuggestedZones() {
        // Рисуем предложенные зоны
        this.redrawZones();
    }

    selectZoneTool(zoneType) {
        this.currentTool = zoneType;
        this.currentZone = [];
        this.isDrawing = false;

        document.querySelectorAll('.zone-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-zone="${zoneType}"]`)?.classList.add('active');

        const instructions = document.getElementById('zoneInstructions');
        if (instructions) instructions.style.display = 'block';

        this.showStatus(`Корректировка зоны: ${this.getZoneLabel(zoneType)}`, 'info');
    }

    handleCanvasClick(event) {
        if (!this.currentTool) return;

        const canvas = event.target;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const videoElement = document.getElementById('videoFrame') || document.querySelector('img');
        if (!videoElement) return;

        const scaleX = videoElement.naturalWidth / rect.width;
        const scaleY = videoElement.naturalHeight / rect.height;

        const imgX = Math.round(x * scaleX);
        const imgY = Math.round(y * scaleY);

        this.currentZone.push([imgX, imgY]);
        this.isDrawing = true;

        this.redrawZones();
    }

    finishZone(event) {
        event.preventDefault();

        if (!this.isDrawing || this.currentZone.length < 3) return;

        this.finalZones[this.currentTool] = [...this.currentZone];
        this.currentZone = [];
        this.isDrawing = false;
        this.currentTool = null;

        document.querySelectorAll('.zone-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('zoneInstructions').style.display = 'none';

        this.redrawZones();
        this.showStatus('Зона обновлена', 'success');
    }

    cancelDrawing() {
        this.currentZone = [];
        this.isDrawing = false;
        this.currentTool = null;

        document.querySelectorAll('.zone-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('zoneInstructions').style.display = 'none';

        this.redrawZones();
    }

    async finalizeZones() {
        try {
            const response = await fetch('/api/intelligent/finalize_zones', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({zones: this.finalZones})
            });

            const data = await response.json();
            if (data.success) {
                this.showStep('learning');
                this.startLearningProcess();
                this.showStatus('Зоны зафиксированы. Начинаем обучение...', 'success');
            } else {
                this.showStatus('Ошибка: ' + data.error, 'error');
            }
        } catch (error) {
            this.showStatus('Ошибка: ' + error.message, 'error');
        }
    }

    async startLearningProcess() {
        // Запускаем периодическое получение объектов для классификации
        this.learningInterval = setInterval(() => {
            this.getNextObject();
        }, 1000);

        this.updateProgress();
    }

    async getNextObject() {
        try {
            const response = await fetch('/api/intelligent/get_next_object');
            const data = await response.json();

            if (data.success && data.object) {
                this.currentObject = data.object;
                this.showObjectForClassification(data.object);
            } else {
                // Нет больше объектов
                clearInterval(this.learningInterval);
                this.showStep('completed');
                this.showCompletionStats();
            }
        } catch (error) {
            console.error('Ошибка получения объекта:', error);
        }
    }

    showObjectForClassification(objectData) {
        const classificationDiv = document.getElementById('objectClassification');
        const imageDiv = document.getElementById('objectImage');
        const infoDiv = document.getElementById('objectInfo');

        if (classificationDiv) classificationDiv.style.display = 'block';

        if (imageDiv) {
            imageDiv.innerHTML = `
                <img src="data:image/jpeg;base64,${objectData.roi_image}" 
                     style="max-width: 200px; max-height: 150px; border: 2px solid #27ae60; border-radius: 8px;">
            `;
        }

        if (infoDiv) {
            const chars = objectData.characteristics;
            infoDiv.innerHTML = `
                <strong>Характеристики объекта:</strong><br>
                • Площадь: ${chars.area} пикселей<br>
                • Размеры: ${chars.dimensions}<br>
                • Соотношение сторон: ${chars.aspect_ratio}<br>
                • Осталось объектов: ${objectData.remaining_objects}
            `;
        }
    }

    showNewProductForm() {
        document.getElementById('newProductForm').style.display = 'block';
        document.getElementById('existingProductsList').style.display = 'none';
    }

    hideNewProductForm() {
        document.getElementById('newProductForm').style.display = 'none';
    }

    async saveNewProduct() {
        const name = document.getElementById('productName')?.value;
        const sku = document.getElementById('productSKU')?.value;
        const weight = document.getElementById('productWeight')?.value;
        const description = document.getElementById('productDescription')?.value;

        if (!name || !sku) {
            this.showStatus('Введите название и SKU код', 'warning');
            return;
        }

        const classification = {
            type: 'new_product',
            product_info: {
                name: name,
                sku_code: sku,
                weight: weight,
                description: description
            }
        };

        await this.submitClassification(classification);
        this.hideNewProductForm();
    }

    async showExistingProducts() {
        // Здесь можно загрузить список существующих продуктов
        document.getElementById('existingProductsList').style.display = 'block';
        document.getElementById('newProductForm').style.display = 'none';
    }

    async classifyAsDefective() {
        const classification = {
            type: 'defective',
            quality_score: 0.0
        };
        await this.submitClassification(classification);
    }

    async classifyAsNotProduct() {
        const classification = {
            type: 'not_product'
        };
        await this.submitClassification(classification);
    }

    async submitClassification(classification) {
        try {
            const response = await fetch('/api/intelligent/classify_object', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    object_data: this.currentObject.object_data,
                    classification: classification
                })
            });

            const data = await response.json();
            if (data.success) {
                this.showStatus('Объект классифицирован', 'success');
                this.hideClassificationUI();
                this.updateProgress();
            } else {
                this.showStatus('Ошибка классификации: ' + data.error, 'error');
            }
        } catch (error) {
            this.showStatus('Ошибка: ' + error.message, 'error');
        }
    }

    hideClassificationUI() {
        document.getElementById('objectClassification').style.display = 'none';
        document.getElementById('newProductForm').style.display = 'none';
        document.getElementById('existingProductsList').style.display = 'none';

        // Очищаем поля формы
        ['productName', 'productSKU', 'productWeight', 'productDescription'].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.value = '';
        });
    }

    async updateProgress() {
        try {
            const response = await fetch('/api/intelligent/get_progress');
            const data = await response.json();

            if (data.success) {
                const progress = data.progress;
                const progressDiv = document.getElementById('learningProgress');

                if (progressDiv) {
                    progressDiv.innerHTML = `
                        <h5>📊 Прогресс обучения:</h5>
                        <p>• Обработано объектов: ${progress.identified_objects}</p>
                        <p>• Взаимодействий с пользователем: ${progress.user_interactions}</p>
                        <p>• Изучено продуктов: ${progress.products_learned}</p>
                        <p>• Статус: ${progress.status}</p>
                    `;
                }
            }
        } catch (error) {
            console.error('Ошибка обновления прогресса:', error);
        }
    }

    async showCompletionStats() {
        try {
            const response = await fetch('/api/intelligent/get_progress');
            const data = await response.json();

            if (data.success) {
                const progress = data.progress;
                const statsDiv = document.getElementById('completionStats');

                if (statsDiv) {
                    let productsHtml = '';
                    progress.products.forEach(product => {
                        productsHtml += `
                            <div style="background: rgba(44, 62, 80, 0.8); padding: 8px; margin: 5px 0; border-radius: 4px;">
                                <strong>${product.name}</strong> (${product.sku_code})<br>
                                <small>Вес: ${product.weight} кг • Образцов: ${product.samples_count}</small>
                            </div>
                        `;
                    });

                    statsDiv.innerHTML = `
                        <h5>🎉 Обучение завершено успешно!</h5>
                        <p><strong>Результаты:</strong></p>
                        <p>• Изучено продуктов: ${progress.products_learned}</p>
                        <p>• Всего образцов: ${progress.identified_objects}</p>
                        <p>• Взаимодействий: ${progress.user_interactions}</p>
                        <br>
                        <h6>📦 Изученные продукты:</h6>
                        ${productsHtml}
                    `;
                }
            }
        } catch (error) {
            console.error('Ошибка получения статистики:', error);
        }
    }

    async generateFinalDataset() {
        this.showStatus('Генерация финального датасета...', 'info');

        try {
            const response = await fetch('/api/intelligent/generate_dataset', {
                method: 'POST'
            });

            const data = await response.json();
            if (data.success) {
                this.showStatus(`Датасет создан: ${data.generated_samples} образцов, ${data.products_count} продуктов`, 'success');
            } else {
                this.showStatus('Ошибка создания датасета: ' + data.error, 'error');
            }
        } catch (error) {
            this.showStatus('Ошибка: ' + error.message, 'error');
        }
    }

    redrawZones() {
        const canvas = document.getElementById('drawingCanvas') || document.querySelector('canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const videoElement = document.getElementById('videoFrame') || document.querySelector('img');
        if (!videoElement) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = rect.width / videoElement.naturalWidth;
        const scaleY = rect.height / videoElement.naturalHeight;

        const zoneColors = {
            'counting_zone': 'rgba(39, 174, 96, 0.3)',
            'entry_zone': 'rgba(52, 152, 219, 0.3)',
            'exit_zone': 'rgba(231, 76, 60, 0.3)'
        };

        // Рисуем финальные зоны
        Object.entries(this.finalZones).forEach(([zoneName, zone]) => {
            if (zone && zoneColors[zoneName]) {
                this.drawZonePolygon(ctx, zone, zoneColors[zoneName], scaleX, scaleY);
            }
        });

        // Рисуем текущую зону в процессе создания
        if (this.currentZone.length > 0) {
            this.drawZonePolygon(ctx, this.currentZone, 'rgba(255, 255, 0, 0.5)', scaleX, scaleY);
        }
    }

    drawZonePolygon(ctx, zone, color, scaleX, scaleY) {
        if (!zone || zone.length < 2) return;

        ctx.fillStyle = color;
        ctx.strokeStyle = color.replace('0.3', '1.0');
        ctx.lineWidth = 2;

        ctx.beginPath();
        const firstPoint = zone[0];
        ctx.moveTo(firstPoint[0] * scaleX, firstPoint[1] * scaleY);

        for (let i = 1; i < zone.length; i++) {
            const point = zone[i];
            ctx.lineTo(point[0] * scaleX, point[1] * scaleY);
        }

        if (zone.length > 2) {
            ctx.closePath();
            ctx.fill();
        }
        ctx.stroke();
    }

    getZoneLabel(zoneType) {
        const labels = {
            counting_zone: 'Зона подсчета',
            entry_zone: 'Зона входа',
            exit_zone: 'Зона выхода'
        };
        return labels[zoneType] || zoneType;
    }

    showStatus(message, type = 'info') {
        if (window.showStatus) {
            window.showStatus(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Автоинициализация
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.intelligentTraining === 'undefined') {
        window.intelligentTraining = new IntelligentTrainingInterface();

        setTimeout(() => {
            window.intelligentTraining.initializeInterface();
        }, 1000);
    }
});
'''

if __name__ == '__main__':
    # Пример использования
    training_manager = IntelligentTrainingManager()

    print("🤖 Интеллектуальная система обучения инициализирована")
    print("🎯 Алгоритм обучения:")
    print("   1. Загрузка видео → автопредложение зон")
    print("   2. Корректировка зон пользователем")
    print("   3. Система находит объекты → пользователь описывает")
    print("   4. Интерактивные вопросы системы")
    print("   5. Создание продуктов 'на лету'")
    print("   6. Генерация умного датасета")

    print("\n🔧 Для интеграции:")
    print(
        "   from intelligent_training_system import IntelligentTrainingManager, add_intelligent_training_routes, INTELLIGENT_TRAINING_JS")

    training_manager.close()