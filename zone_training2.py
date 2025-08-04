# zone_training_interface.py - Полная система зонной разметки для производства
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import base64
from werkzeug.utils import secure_filename
import threading
import queue


class ZoneTrainingApp:
    """Система зонной разметки для реального производства хлеба"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Настройки
        self.app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB
        self.app.config['UPLOAD_FOLDER'] = 'uploads'

        # Создаем папки
        for folder in ['uploads', 'training_data/images', 'training_data/annotations',
                       'training_data/zones', 'training_data/models']:
            os.makedirs(folder, exist_ok=True)

        # Состояние видео
        self.current_video = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.video_cap = None
        self.fps = 25.0

        # Зоны (сохраняются в координатах изображения)
        self.zones = {
            'counting_zone': None,  # Основная зона подсчета
            'entry_zone': None,  # Зона входа (хлеб выходит из печи)
            'exit_zone': None,  # Зона выхода (хлеб идет на стол)
            'exclude_zones': []  # Зоны исключения (края, препятствия)
        }

        # Информация о партии
        self.current_batch = {
            'name': '',
            'weight': 0.0,
            'target_count': 0,
            'bread_type': 'standard'
        }

        # Настройки детекции
        self.detection_params = {
            'min_area': 2000,
            'max_area': 25000,
            'hsv_lower': [10, 20, 20],
            'hsv_upper': [30, 255, 200]
        }

        # Статистика
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'valid_detections': 0
        }

        self._setup_routes()

    def _setup_routes(self):
        """Настройка всех маршрутов Flask"""

        @self.app.route('/')
        def main_interface():
            return render_template_string(self._get_main_template())

        @self.app.route('/upload_video', methods=['POST'])
        def upload_video():
            """Загрузка видео файла"""
            try:
                if 'video' not in request.files:
                    return jsonify({'success': False, 'error': 'Нет файла'})

                file = request.files['video']
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'Файл не выбран'})

                if file and self._allowed_video_file(file.filename):
                    filename = secure_filename(file.filename)
                    # Добавляем timestamp для уникальности
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{int(time.time())}{ext}"

                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    # Загружаем видео
                    if self._load_video(filepath):
                        return jsonify({
                            'success': True,
                            'filename': filename,
                            'total_frames': self.total_frames,
                            'fps': self.fps
                        })
                    else:
                        os.remove(filepath)
                        return jsonify({'success': False, 'error': 'Не удалось загрузить видео'})

                return jsonify({'success': False, 'error': 'Неподдерживаемый формат'})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/list_videos')
        def list_videos():
            """Список загруженных видео"""
            try:
                videos = []
                upload_dir = self.app.config['UPLOAD_FOLDER']

                if os.path.exists(upload_dir):
                    for filename in os.listdir(upload_dir):
                        if self._allowed_video_file(filename):
                            filepath = os.path.join(upload_dir, filename)
                            size = os.path.getsize(filepath)
                            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                            videos.append({
                                'filename': filename,
                                'size': self._format_file_size(size),
                                'modified': mtime.strftime('%d.%m.%Y %H:%M'),
                                'has_zones': self._video_has_zones(filename)
                            })

                return jsonify({'videos': sorted(videos, key=lambda x: x['filename'])})

            except Exception as e:
                return jsonify({'videos': [], 'error': str(e)})

        @self.app.route('/load_video', methods=['POST'])
        def load_video():
            """Загрузка выбранного видео"""
            try:
                data = request.get_json()
                filename = data.get('filename')

                if not filename:
                    return jsonify({'success': False, 'error': 'Не указан файл'})

                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

                if not os.path.exists(filepath):
                    return jsonify({'success': False, 'error': 'Файл не найден'})

                if self._load_video(filepath):
                    # Загружаем сохраненные зоны
                    self._load_zones_for_video(filename)

                    return jsonify({
                        'success': True,
                        'total_frames': self.total_frames,
                        'fps': self.fps,
                        'zones': self.zones,
                        'batch': self.current_batch
                    })
                else:
                    return jsonify({'success': False, 'error': 'Ошибка загрузки видео'})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/get_frame', methods=['POST'])
        def get_frame():
            """Получение кадра с детекцией и зонами"""
            try:
                data = request.get_json()
                frame_index = int(data.get('frame_index', 0))

                if not self.video_cap:
                    return jsonify({'success': False, 'error': 'Видео не загружено'})

                frame_data = self._get_frame_with_detections(frame_index)

                if frame_data:
                    return jsonify({
                        'success': True,
                        'frame_data': frame_data
                    })
                else:
                    return jsonify({'success': False, 'error': 'Ошибка получения кадра'})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/save_zones', methods=['POST'])
        def save_zones():
            """Сохранение зон"""
            try:
                data = request.get_json()
                self.zones = data.get('zones', {})

                # Сохраняем в файл
                zones_file = self._get_zones_filename()
                os.makedirs(os.path.dirname(zones_file), exist_ok=True)

                save_data = {
                    'zones': self.zones,
                    'batch': self.current_batch,
                    'detection_params': self.detection_params,
                    'video_info': {
                        'filename': os.path.basename(self.current_video) if self.current_video else '',
                        'total_frames': self.total_frames,
                        'fps': self.fps
                    },
                    'created': datetime.now().isoformat()
                }

                with open(zones_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)

                return jsonify({'success': True, 'message': 'Зоны сохранены'})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/set_batch_info', methods=['POST'])
        def set_batch_info():
            """Установка информации о партии"""
            try:
                data = request.get_json()
                self.current_batch.update(data)

                return jsonify({'success': True, 'batch': self.current_batch})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/generate_training_data', methods=['POST'])
        def generate_training_data():
            """Генерация обучающих данных"""
            try:
                data = request.get_json()
                frames_count = int(data.get('frames_count', 100))

                if not self.zones.get('counting_zone'):
                    return jsonify({'success': False, 'error': 'Не определена зона подсчета'})

                # Генерируем данные
                result = self._create_training_dataset(frames_count)

                return jsonify({
                    'success': True,
                    'generated': result['generated'],
                    'total_objects': result['total_objects'],
                    'dataset_path': result['dataset_path']
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/get_stats')
        def get_stats():
            """Получение статистики"""
            return jsonify({
                'stats': self.stats,
                'zones_count': len([z for z in self.zones.values() if z]),
                'video_loaded': self.current_video is not None,
                'batch': self.current_batch
            })

    def _allowed_video_file(self, filename):
        """Проверка разрешенных форматов видео"""
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    def _format_file_size(self, size_bytes):
        """Форматирование размера файла"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def _video_has_zones(self, filename):
        """Проверка наличия зон для видео"""
        zones_file = self._get_zones_filename_for_video(filename)
        return os.path.exists(zones_file)

    def _get_zones_filename_for_video(self, filename):
        """Получение имени файла зон для конкретного видео"""
        video_name = Path(filename).stem
        return f"training_data/zones/{video_name}_zones.json"

    def _load_video(self, filepath):
        """Загрузка видео файла"""
        try:
            self.video_cap = cv2.VideoCapture(filepath)

            if not self.video_cap.isOpened():
                return False

            self.current_video = filepath
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.current_frame_index = 0

            # Проверяем первый кадр
            ret, frame = self.video_cap.read()
            if ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"Видео загружено: {filepath}")
                print(f"Кадров: {self.total_frames}, FPS: {self.fps}")
                return True

            return False

        except Exception as e:
            print(f"Ошибка загрузки видео: {e}")
            return False

    def _get_frame_with_detections(self, frame_index):
        """Получение кадра с детекцией и зонами"""
        try:
            # Устанавливаем позицию кадра
            self.video_cap.set(cv2.CAV_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_cap.read()

            if not ret:
                return None

            self.current_frame_index = frame_index

            # Детекция объектов
            detections = self._detect_bread_objects(frame)

            # Создаем аннотированный кадр
            annotated_frame = self._draw_zones_and_objects(frame.copy(), detections)

            # Конвертируем в base64
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Обновляем статистику
            self.stats['frames_processed'] += 1
            self.stats['total_detections'] += len(detections)
            self.stats['valid_detections'] += len([d for d in detections if d['in_counting_zone']])

            return {
                'frame_index': frame_index,
                'image': img_base64,
                'detections': detections,
                'zones': self.zones,
                'timestamp': frame_index / self.fps,
                'stats': {
                    'total_objects': len(detections),
                    'in_counting_zone': len([d for d in detections if d['in_counting_zone']]),
                    'in_entry_zone': len([d for d in detections if d['in_entry_zone']]),
                    'in_exit_zone': len([d for d in detections if d['in_exit_zone']])
                }
            }

        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
            return None

    def _detect_bread_objects(self, frame):
        """Детекция хлебных буханок"""
        detections = []

        # Преобразуем в HSV для лучшей детекции хлебных оттенков
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Создаем маску для хлебных цветов
        lower = np.array(self.detection_params['hsv_lower'])
        upper = np.array(self.detection_params['hsv_upper'])
        mask = cv2.inRange(hsv, lower, upper)

        # Морфологические операции для улучшения маски
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Размытие для сглаживания
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Фильтрация по размеру
            if area < self.detection_params['min_area'] or area > self.detection_params['max_area']:
                continue

            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2

            # Проверяем попадание в зоны исключения
            if self._point_in_exclude_zones(center_x, center_y):
                continue

            # Вычисляем дополнительные характеристики
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0

            # Фильтрация по форме (хлеб должен быть относительно овальным)
            if aspect_ratio < 0.5 or aspect_ratio > 3.0 or extent < 0.4:
                continue

            detection = {
                'id': i,
                'bbox': [x, y, x + w, y + h],
                'center': [center_x, center_y],
                'area': area,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'confidence': min(0.95, 0.5 + extent * 0.5),  # Простая оценка уверенности
                'in_counting_zone': self._point_in_zone(center_x, center_y, 'counting_zone'),
                'in_entry_zone': self._point_in_zone(center_x, center_y, 'entry_zone'),
                'in_exit_zone': self._point_in_zone(center_x, center_y, 'exit_zone')
            }

            detections.append(detection)

        return detections

    def _point_in_zone(self, x, y, zone_name):
        """Проверка попадания точки в зону"""
        zone = self.zones.get(zone_name)
        if not zone or len(zone) < 3:
            return False

        points = np.array(zone, np.int32)
        return cv2.pointPolygonTest(points, (x, y), False) >= 0

    def _point_in_exclude_zones(self, x, y):
        """Проверка попадания точки в зоны исключения"""
        for zone in self.zones.get('exclude_zones', []):
            if zone and len(zone) >= 3:
                points = np.array(zone, np.int32)
                if cv2.pointPolygonTest(points, (x, y), False) >= 0:
                    return True
        return False

    def _draw_zones_and_objects(self, frame, detections):
        """Отрисовка зон и детектированных объектов"""

        # Цвета для зон
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
                        self._draw_zone(frame, zone, color)
            else:
                zone = self.zones.get(zone_name)
                if zone:
                    self._draw_zone(frame, zone, color)

        # Рисуем детекции
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            center_x, center_y = detection['center']

            # Выбираем цвет в зависимости от зоны
            if detection['in_counting_zone']:
                color = (0, 255, 0)  # Зеленый - в зоне подсчета
                thickness = 3
            elif detection['in_entry_zone']:
                color = (255, 0, 0)  # Синий - в зоне входа
                thickness = 2
            elif detection['in_exit_zone']:
                color = (0, 0, 255)  # Красный - в зоне выхода
                thickness = 2
            else:
                color = (255, 255, 255)  # Белый - вне основных зон
                thickness = 1

            # Рисуем bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # ID и информация
            label = f"ID:{detection['id']} ({detection['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Центр объекта
            cv2.circle(frame, (center_x, center_y), 3, color, -1)

        # Добавляем информацию о партии
        if self.current_batch['name']:
            info_text = f"Партия: {self.current_batch['name']}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Статистика на кадре
        stats_text = f"Кадр: {self.current_frame_index}/{self.total_frames}"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _draw_zone(self, frame, zone, color):
        """Отрисовка зоны"""
        if not zone or len(zone) < 3:
            return

        points = np.array(zone, np.int32)

        # Полупрозрачная заливка
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Контур зоны
        cv2.polylines(frame, [points], True, color, 2)

    def _load_zones_for_video(self, filename):
        """Загрузка сохраненных зон для видео"""
        zones_file = self._get_zones_filename_for_video(filename)
        if os.path.exists(zones_file):
            try:
                with open(zones_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.zones = data.get('zones', self.zones)
                    self.current_batch = data.get('batch', self.current_batch)
                    self.detection_params = data.get('detection_params', self.detection_params)
                print(f"Зоны загружены для {filename}")
            except Exception as e:
                print(f"Ошибка загрузки зон: {e}")

    def _get_zones_filename(self):
        """Получение имени файла зон для текущего видео"""
        if self.current_video:
            video_name = Path(self.current_video).stem
            return f"training_data/zones/{video_name}_zones.json"
        return "training_data/zones/default_zones.json"

    def _create_training_dataset(self, frames_count):
        """Создание датасета для обучения"""
        if not self.video_cap:
            return {'generated': 0, 'total_objects': 0, 'dataset_path': ''}

        # Создаем папку для датасета
        dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = f"training_data/{dataset_name}"

        os.makedirs(f"{dataset_path}/images", exist_ok=True)
        os.makedirs(f"{dataset_path}/annotations", exist_ok=True)

        generated_count = 0
        total_objects = 0

        # Выбираем кадры равномерно по всему видео
        step = max(1, self.total_frames // frames_count)

        for frame_idx in range(0, self.total_frames, step):
            if generated_count >= frames_count:
                break

            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()

            if not ret:
                continue

            # Детекция объектов
            detections = self._detect_bread_objects(frame)

            # Фильтруем только объекты в зоне подсчета
            valid_detections = [d for d in detections if d['in_counting_zone']]

            if len(valid_detections) == 0:
                continue

            # Сохраняем изображение
            img_filename = f"frame_{frame_idx:06d}.jpg"
            img_path = f"{dataset_path}/images/{img_filename}"
            cv2.imwrite(img_path, frame)

            # Создаем аннотацию в формате YOLO
            ann_filename = f"frame_{frame_idx:06d}.txt"
            ann_path = f"{dataset_path}/annotations/{ann_filename}"

            h, w = frame.shape[:2]

            with open(ann_path, 'w') as f:
                for detection in valid_detections:
                    x1, y1, x2, y2 = detection['bbox']

                    # Нормализуем координаты для YOLO
                    center_x = ((x1 + x2) / 2) / w
                    center_y = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    # Класс 0 для хлеба
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

            generated_count += 1
            total_objects += len(valid_detections)

        # Сохраняем метаданные датасета
        metadata = {
            'created': datetime.now().isoformat(),
            'video_source': os.path.basename(self.current_video),
            'batch_info': self.current_batch,
            'zones': self.zones,
            'detection_params': self.detection_params,
            'frames_generated': generated_count,
            'total_objects': total_objects,
            'classes': ['bread']
        }

        with open(f"{dataset_path}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Создаем data.yaml для YOLO
        yaml_content = f"""train: {dataset_path}/images
val: {dataset_path}/images
nc: 1
names: ['bread']
"""
        with open(f"{dataset_path}/data.yaml", 'w') as f:
            f.write(yaml_content)

        return {
            'generated': generated_count,
            'total_objects': total_objects,
            'dataset_path': dataset_path
        }

    def _get_main_template(self):
        """HTML шаблон главного интерфейса"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Система зонной разметки - Производство хлеба</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            overflow-x: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            padding: 15px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .header p {
            color: #bdc3c7;
            font-size: 14px;
        }

        .main-container {
            display: flex;
            height: calc(100vh - 80px);
        }

        .video-section {
            flex: 1;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }

        .control-panel {
            width: 350px;
            background: #2c3e50;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #34495e;
        }

        .video-upload {
            background: #34495e;
            border: 2px dashed #52c234;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .video-upload:hover {
            background: #3d566e;
            border-color: #6dd646;
        }

        .video-upload.dragover {
            background: #52c234;
            color: #1a1a1a;
        }

        .video-container {
            position: relative;
            flex: 1;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            display: none;
        }

        .video-container.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #videoFrame {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        #drawingCanvas {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
        }

        .video-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 15px;
            background: #34495e;
            border-radius: 10px;
            margin-top: 10px;
        }

        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .btn:hover {
            background: #2980b9;
            transform: translateY(-1px);
        }

        .btn.success {
            background: #27ae60;
        }

        .btn.success:hover {
            background: #229954;
        }

        .btn.danger {
            background: #e74c3c;
        }

        .btn.danger:hover {
            background: #c0392b;
        }

        .btn.secondary {
            background: #95a5a6;
        }

        .btn.secondary:hover {
            background: #7f8c8d;
        }

        .frame-slider {
            flex: 1;
            height: 30px;
        }

        .zone-tools {
            margin-bottom: 20px;
        }

        .zone-tool-btn {
            display: block;
            width: 100%;
            margin-bottom: 10px;
            padding: 12px;
            background: #34495e;
            border: 2px solid transparent;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: left;
        }

        .zone-tool-btn:hover {
            background: #4a6582;
        }

        .zone-tool-btn.active {
            border-color: #3498db;
            background: #2980b9;
        }

        .batch-info {
            background: #34495e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .batch-input {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #52c234;
            border-radius: 4px;
            background: #2c3e50;
            color: white;
        }

        .zone-legend {
            background: #34495e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .zone-legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }

        .zone-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 10px;
            border: 1px solid #666;
        }

        .stats-panel {
            background: #34495e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .status-message {
            position: fixed;
            top: 100px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }

        .status-message.show {
            transform: translateX(0);
        }

        .status-message.success {
            background: #27ae60;
        }

        .status-message.error {
            background: #e74c3c;
        }

        .status-message.warning {
            background: #f39c12;
        }

        .status-message.info {
            background: #3498db;
        }

        .hidden {
            display: none !important;
        }

        .video-list {
            background: #34495e;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }

        .video-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #2c3e50;
            border-radius: 5px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .video-item:hover {
            background: #3d566e;
        }

        .video-item.has-zones {
            border-left: 4px solid #27ae60;
        }

        #drawingInstructions {
            background: #f39c12;
            color: #1a1a1a;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            display: none;
        }

        #drawingInstructions.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 Система зонной разметки - Производство хлеба</h1>
        <p>Разметка зон для автоматического подсчета и трекинга буханок хлеба на производственной линии</p>
    </div>

    <div class="main-container">
        <div class="video-section">
            <!-- Загрузка видео -->
            <div class="video-upload" id="videoUpload">
                <h3>📂 Загрузите видео производственной линии</h3>
                <p>Поддерживаемые форматы: MP4, AVI, MOV, MKV, WebM</p>
                <input type="file" id="videoFile" accept="video/*" style="display: none;">
                <p style="margin-top: 10px;">
                    <button class="btn success" onclick="document.getElementById('videoFile').click()">
                        📂 Выбрать файл
                    </button>
                    <button class="btn secondary" id="showVideoList">
                        📋 Загруженные видео
                    </button>
                </p>
            </div>

            <!-- Список видео -->
            <div class="video-list hidden" id="videoList">
                <h4>📋 Загруженные видео:</h4>
                <div id="videoItems"></div>
                <button class="btn secondary" id="hideVideoList">❌ Скрыть</button>
            </div>

            <!-- Контейнер видео -->
            <div class="video-container" id="videoContainer">
                <img id="videoFrame" alt="Кадр видео">
                <canvas id="drawingCanvas"></canvas>
            </div>

            <!-- Управление видео -->
            <div class="video-controls hidden" id="videoControls">
                <button class="btn" id="prevFrame">⏮️</button>
                <input type="range" id="frameSlider" class="frame-slider" min="0" max="100" value="0">
                <button class="btn" id="nextFrame">⏭️</button>
                <span id="frameInfo">0 / 0</span>
                <button class="btn success" id="autoDetect">🔍 Детекция</button>
            </div>
        </div>

        <div class="control-panel">
            <h3>🛠️ Инструменты зонирования</h3>

            <!-- Инструкции по рисованию -->
            <div id="drawingInstructions">
                <strong>Рисование зоны:</strong><br>
                • Кликайте мышкой по углам зоны<br>
                • Двойной клик - завершить зону<br>
                • ESC - отменить рисование
            </div>

            <!-- Инструменты зон -->
            <div class="zone-tools">
                <h4>Инструменты рисования:</h4>
                <button class="zone-tool-btn" data-tool="counting_zone">
                    🟢 Зона подсчета
                    <small style="display: block; color: #bdc3c7; margin-top: 5px;">
                        Основная зона для подсчета готового хлеба
                    </small>
                </button>
                <button class="zone-tool-btn" data-tool="entry_zone">
                    🔵 Зона входа
                    <small style="display: block; color: #bdc3c7; margin-top: 5px;">
                        Зона где хлеб выходит из печи
                    </small>
                </button>
                <button class="zone-tool-btn" data-tool="exit_zone">
                    🔴 Зона выхода
                    <small style="display: block; color: #bdc3c7; margin-top: 5px;">
                        Зона где хлеб уходит на стол/конвейер
                    </small>
                </button>
                <button class="zone-tool-btn" data-tool="exclude_zone">
                    ⚫ Зона исключения
                    <small style="display: block; color: #bdc3c7; margin-top: 5px;">
                        Область которую нужно игнорировать
                    </small>
                </button>
                <button class="zone-tool-btn" data-tool="edit">
                    ✏️ Редактировать зоны
                    <small style="display: block; color: #bdc3c7; margin-top: 5px;">
                        Изменение существующих зон
                    </small>
                </button>
            </div>

            <!-- Информация о партии -->
            <div class="batch-info">
                <h4>📦 Информация о партии:</h4>
                <input type="text" id="batchName" class="batch-input" 
                       placeholder="Название хлеба (напр: Олександрівський)">
                <input type="number" id="batchWeight" class="batch-input" 
                       placeholder="Вес буханки (кг)" step="0.1" min="0.1" max="2.0">
                <input type="number" id="targetCount" class="batch-input" 
                       placeholder="Ожидаемое количество" min="1">
                <button class="btn success" id="saveBatchInfo">💾 Сохранить партию</button>
            </div>

            <!-- Легенда зон -->
            <div class="zone-legend">
                <h4>🎨 Легенда зон:</h4>
                <div class="zone-legend-item">
                    <div class="zone-color" style="background: rgba(0, 255, 0, 0.7);"></div>
                    <span>Зона подсчета</span>
                </div>
                <div class="zone-legend-item">
                    <div class="zone-color" style="background: rgba(255, 0, 0, 0.7);"></div>
                    <span>Зона входа</span>
                </div>
                <div class="zone-legend-item">
                    <div class="zone-color" style="background: rgba(0, 0, 255, 0.7);"></div>
                    <span>Зона выхода</span>
                </div>
                <div class="zone-legend-item">
                    <div class="zone-color" style="background: rgba(128, 128, 128, 0.7);"></div>
                    <span>Исключение</span>
                </div>
            </div>

            <!-- Статистика -->
            <div class="stats-panel">
                <h4>📊 Статистика:</h4>
                <div id="statsContent">
                    <p>Загрузите видео для начала работы</p>
                </div>
            </div>

            <!-- Управление -->
            <div style="text-align: center;">
                <button class="btn success" id="saveZones">💾 Сохранить зоны</button>
                <button class="btn secondary" id="clearZones">🗑️ Очистить все</button>
                <button class="btn success" id="generateDataset" style="margin-top: 10px;">
                    🚀 Создать датасет
                </button>
                <input type="number" id="framesCount" placeholder="Кол-во кадров" 
                       value="200" min="10" max="1000" style="margin-top: 10px; width: 100%;" class="batch-input">
            </div>
        </div>
    </div>

    <!-- Сообщения о статусе -->
    <div id="statusMessage" class="status-message"></div>

    <script>
        // Глобальные переменные
        let currentFrame = 0;
        let totalFrames = 0;
        let isDrawing = false;
        let currentTool = null;
        let currentZone = [];
        let zones = {
            counting_zone: null,
            entry_zone: null,
            exit_zone: null,
            exclude_zones: []
        };

        // DOM элементы
        const videoUpload = document.getElementById('videoUpload');
        const videoFile = document.getElementById('videoFile');
        const videoContainer = document.getElementById('videoContainer');
        const videoFrame = document.getElementById('videoFrame');
        const drawingCanvas = document.getElementById('drawingCanvas');
        const videoControls = document.getElementById('videoControls');
        const frameSlider = document.getElementById('frameSlider');
        const frameInfo = document.getElementById('frameInfo');
        const drawingInstructions = document.getElementById('drawingInstructions');

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            initializeEventListeners();
            loadVideoList();
            updateStats();
        });

        function initializeEventListeners() {
            // Загрузка видео
            videoFile.addEventListener('change', handleVideoUpload);

            // Drag & Drop
            videoUpload.addEventListener('dragover', handleDragOver);
            videoUpload.addEventListener('drop', handleDrop);

            // Управление кадрами
            document.getElementById('prevFrame').addEventListener('click', () => changeFrame(-1));
            document.getElementById('nextFrame').addEventListener('click', () => changeFrame(1));
            frameSlider.addEventListener('input', (e) => loadFrame(parseInt(e.target.value)));

            // Инструменты зон
            document.querySelectorAll('.zone-tool-btn').forEach(btn => {
                btn.addEventListener('click', (e) => selectTool(e.target.dataset.tool));
            });

            // Canvas для рисования
            drawingCanvas.addEventListener('click', handleCanvasClick);
            drawingCanvas.addEventListener('dblclick', finishZone);

            // Клавиатура
            document.addEventListener('keydown', handleKeyPress);

            // Кнопки управления
            document.getElementById('saveZones').addEventListener('click', saveZones);
            document.getElementById('clearZones').addEventListener('click', clearAllZones);
            document.getElementById('saveBatchInfo').addEventListener('click', saveBatchInfo);
            document.getElementById('generateDataset').addEventListener('click', generateDataset);
            document.getElementById('autoDetect').addEventListener('click', toggleAutoDetection);

            // Список видео
            document.getElementById('showVideoList').addEventListener('click', showVideoList);
            document.getElementById('hideVideoList').addEventListener('click', hideVideoList);
        }

        function handleVideoUpload(event) {
            const file = event.target.files[0];
            if (file) {
                uploadVideo(file);
            }
        }

        function handleDragOver(event) {
            event.preventDefault();
            videoUpload.classList.add('dragover');
        }

        function handleDrop(event) {
            event.preventDefault();
            videoUpload.classList.remove('dragover');

            const files = event.dataTransfer.files;
            if (files.length > 0) {
                uploadVideo(files[0]);
            }
        }

        function uploadVideo(file) {
            if (!file.type.startsWith('video/')) {
                showStatus('Выберите видео файл', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('video', file);

            showStatus('Загрузка видео...', 'info');

            fetch('/upload_video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    totalFrames = data.total_frames;
                    setupVideoInterface();
                    loadFrame(0);
                    showStatus(`Видео загружено: ${data.total_frames} кадров`, 'success');
                } else {
                    showStatus('Ошибка загрузки: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка: ' + error.message, 'error');
            });
        }

        function loadVideoList() {
            fetch('/list_videos')
            .then(response => response.json())
            .then(data => {
                const videoItems = document.getElementById('videoItems');
                videoItems.innerHTML = '';

                if (data.videos.length === 0) {
                    videoItems.innerHTML = '<p>Нет загруженных видео</p>';
                    return;
                }

                data.videos.forEach(video => {
                    const item = document.createElement('div');
                    item.className = 'video-item' + (video.has_zones ? ' has-zones' : '');
                    item.innerHTML = `
                        <div>
                            <strong>${video.filename}</strong><br>
                            <small>${video.size} • ${video.modified}</small>
                            ${video.has_zones ? '<br><small style="color: #27ae60;">✓ Есть зоны</small>' : ''}
                        </div>
                        <button class="btn" onclick="loadExistingVideo('${video.filename}')">
                            Загрузить
                        </button>
                    `;
                    videoItems.appendChild(item);
                });
            });
        }

        function loadExistingVideo(filename) {
            fetch('/load_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: filename})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    totalFrames = data.total_frames;
                    zones = data.zones || zones;

                    // Загружаем информацию о партии
                    if (data.batch) {
                        document.getElementById('batchName').value = data.batch.name || '';
                        document.getElementById('batchWeight').value = data.batch.weight || '';
                        document.getElementById('targetCount').value = data.batch.target_count || '';
                    }

                    setupVideoInterface();
                    loadFrame(0);
                    hideVideoList();
                    showStatus(`Видео загружено: ${filename}`, 'success');
                } else {
                    showStatus('Ошибка: ' + data.error, 'error');
                }
            });
        }

        function setupVideoInterface() {
            videoUpload.style.display = 'none';
            document.getElementById('videoList').classList.add('hidden');
            videoContainer.classList.add('active');
            videoControls.classList.remove('hidden');

            frameSlider.max = totalFrames - 1;
            updateFrameInfo();
        }

        function loadFrame(frameIndex) {
            if (frameIndex < 0 || frameIndex >= totalFrames) return;

            currentFrame = frameIndex;
            frameSlider.value = frameIndex;

            fetch('/get_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frame_index: frameIndex})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const frameData = data.frame_data;
                    videoFrame.src = 'data:image/jpeg;base64,' + frameData.image;

                    // Обновляем статистику
                    updateFrameStats(frameData.stats);
                    updateFrameInfo();

                    // Обновляем размер canvas
                    videoFrame.onload = () => {
                        resizeCanvas();
                    };
                }
            });
        }

        function changeFrame(delta) {
            const newFrame = currentFrame + delta;
            if (newFrame >= 0 && newFrame < totalFrames) {
                loadFrame(newFrame);
            }
        }

        function resizeCanvas() {
            const rect = videoFrame.getBoundingClientRect();
            drawingCanvas.width = rect.width;
            drawingCanvas.height = rect.height;
            drawingCanvas.style.width = rect.width + 'px';
            drawingCanvas.style.height = rect.height + 'px';

            redrawZones();
        }

        function selectTool(tool) {
            // Снимаем выделение с других кнопок
            document.querySelectorAll('.zone-tool-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            // Выделяем текущую кнопку
            document.querySelector(`[data-tool="${tool}"]`).classList.add('active');

            currentTool = tool;
            isDrawing = false;
            currentZone = [];

            if (tool !== 'edit') {
                drawingInstructions.classList.add('show');
            } else {
                drawingInstructions.classList.remove('show');
            }
        }

        function handleCanvasClick(event) {
            if (!currentTool || currentTool === 'edit') return;

            const rect = drawingCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Преобразуем в координаты изображения
            const scaleX = videoFrame.naturalWidth / rect.width;
            const scaleY = videoFrame.naturalHeight / rect.height;

            const imgX = Math.round(x * scaleX);
            const imgY = Math.round(y * scaleY);

            currentZone.push([imgX, imgY]);
            isDrawing = true;

            redrawZones();
        }

        function finishZone(event) {
            event.preventDefault();

            if (!isDrawing || currentZone.length < 3) return;

            if (currentTool === 'exclude_zone') {
                zones.exclude_zones.push([...currentZone]);
            } else {
                zones[currentTool] = [...currentZone];
            }

            currentZone = [];
            isDrawing = false;
            drawingInstructions.classList.remove('show');

            // Снимаем выделение с кнопки
            document.querySelector(`[data-tool="${currentTool}"]`).classList.remove('active');
            currentTool = null;

            redrawZones();
            showStatus('Зона сохранена', 'success');
        }

        function redrawZones() {
            const ctx = drawingCanvas.getContext('2d');
            ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

            const rect = drawingCanvas.getBoundingClientRect();
            const scaleX = rect.width / videoFrame.naturalWidth;
            const scaleY = rect.height / videoFrame.naturalHeight;

            const zoneColors = {
                'counting_zone': 'rgba(0, 255, 0, 0.3)',
                'entry_zone': 'rgba(255, 0, 0, 0.3)',
                'exit_zone': 'rgba(0, 0, 255, 0.3)',
                'exclude_zones': 'rgba(128, 128, 128, 0.3)'
            };

            // Рисуем существующие зоны
            for (const [zoneName, color] of Object.entries(zoneColors)) {
                if (zoneName === 'exclude_zones') {
                    zones.exclude_zones.forEach(zone => {
                        drawZoneOnCanvas(ctx, zone, color, scaleX, scaleY);
                    });
                } else {
                    const zone = zones[zoneName];
                    if (zone) {
                        drawZoneOnCanvas(ctx, zone, color, scaleX, scaleY);
                    }
                }
            }

            // Рисуем текущую зону в процессе создания
            if (currentZone.length > 0) {
                drawZoneOnCanvas(ctx, currentZone, 'rgba(255, 255, 0, 0.5)', scaleX, scaleY);
            }
        }

        function drawZoneOnCanvas(ctx, zone, color, scaleX, scaleY) {
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

            // Рисуем точки
            ctx.fillStyle = color.replace('0.3', '1.0');
            zone.forEach(point => {
                ctx.beginPath();
                ctx.arc(point[0] * scaleX, point[1] * scaleY, 4, 0, 2 * Math.PI);
                ctx.fill();
            });
        }

        function handleKeyPress(event) {
            if (event.key === 'Escape') {
                currentZone = [];
                isDrawing = false;
                currentTool = null;
                drawingInstructions.classList.remove('show');

                document.querySelectorAll('.zone-tool-btn').forEach(btn => {
                    btn.classList.remove('active');
                });

                redrawZones();
            }
        }

        function saveZones() {
            fetch('/save_zones', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({zones: zones})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Зоны сохранены', 'success');
                } else {
                    showStatus('Ошибка сохранения: ' + data.error, 'error');
                }
            });
        }

        function clearAllZones() {
            if (confirm('Удалить все зоны? Это действие нельзя отменить.')) {
                zones = {
                    counting_zone: null,
                    entry_zone: null,
                    exit_zone: null,
                    exclude_zones: []
                };
                redrawZones();
                showStatus('Все зоны очищены', 'info');
            }
        }

        function saveBatchInfo() {
            const batchData = {
                name: document.getElementById('batchName').value.trim(),
                weight: parseFloat(document.getElementById('batchWeight').value) || 0,
                target_count: parseInt(document.getElementById('targetCount').value) || 0
            };

            if (!batchData.name) {
                showStatus('Введите название партии', 'warning');
                return;
            }

            fetch('/set_batch_info', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(batchData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Информация о партии "${batchData.name}" сохранена`, 'success');
                } else {
                    showStatus('Ошибка сохранения', 'error');
                }
            });
        }

        function generateDataset() {
            const framesCount = parseInt(document.getElementById('framesCount').value) || 200;

            if (!zones.counting_zone) {
                showStatus('Создайте сначала зону подсчета', 'warning');
                return;
            }

            showStatus(`Генерация датасета из ${framesCount} кадров...`, 'info');

            fetch('/generate_training_data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frames_count: framesCount})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Датасет создан: ${data.generated} кадров, ${data.total_objects} объектов`, 'success');
                } else {
                    showStatus('Ошибка: ' + data.error, 'error');
                }
            });
        }

        function toggleAutoDetection() {
            loadFrame(currentFrame);
        }

        function showVideoList() {
            document.getElementById('videoList').classList.remove('hidden');
            loadVideoList();
        }

        function hideVideoList() {
            document.getElementById('videoList').classList.add('hidden');
        }

        function updateFrameInfo() {
            frameInfo.textContent = `${currentFrame + 1} / ${totalFrames}`;
        }

        function updateFrameStats(stats) {
            if (stats) {
                document.getElementById('statsContent').innerHTML = `
                    <p><strong>Кадр ${currentFrame + 1}:</strong></p>
                    <p>• Всего объектов: ${stats.total_objects}</p>
                    <p>• В зоне подсчета: ${stats.in_counting_zone}</p>
                    <p>• В зоне входа: ${stats.in_entry_zone}</p>
                    <p>• В зоне выхода: ${stats.in_exit_zone}</p>
                `;
            }
        }

        function updateStats() {
            fetch('/get_stats')
            .then(response => response.json())
            .then(data => {
                // Здесь можно обновить общую статистику
            });
        }

        function showStatus(message, type = 'info') {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = `status-message ${type} show`;

            setTimeout(() => {
                statusEl.classList.remove('show');
            }, 4000);
        }

        // Обновляем размер canvas при изменении размера окна
        window.addEventListener('resize', () => {
            if (videoFrame.complete) {
                resizeCanvas();
            }
        });
    </script>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Запуск приложения"""
        print(f"🏭 Система зонной разметки запущена на http://{host}:{port}")
        print("📋 Возможности системы:")
        print("   • Загрузка видео производственной линии")
        print("   • Рисование зон мышкой (подсчет, вход, выход, исключение)")
        print("   • Автоматическая детекция хлеба в зонах")
        print("   • Генерация обучающих данных для ML")
        print("   • Сохранение настроек для каждого видео")

        self.app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    app = ZoneTrainingApp()
    app.run(debug=True)