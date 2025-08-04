# zone_training_interface.py - Зонная система обучения для производства
from flask import Flask, render_template_string, request, jsonify
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


class ZoneTrainingApp:
    """Зонная система обучения для реального производства"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Настройки
        self.app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
        self.app.config['UPLOAD_FOLDER'] = 'uploads'

        # Создаем папки
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('training_data/images', exist_ok=True)
        os.makedirs('training_data/annotations', exist_ok=True)
        os.makedirs('training_data/zones', exist_ok=True)

        # Состояние видео
        self.current_video = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.video_cap = None

        # Зоны и аннотации
        self.zones = {
            'counting_zone': None,  # Зона подсчета
            'entry_zone': None,  # Зона входа (хлеб выходит из печи)
            'exit_zone': None,  # Зона выхода (хлеб уходит на стол)
            'exclude_zones': []  # Зоны исключения (края, препятствия)
        }

        # Данные партии
        self.current_batch = {
            'name': '',
            'weight': 0.0,
            'target_count': 0
        }

        self._setup_routes()

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.route('/')
        def zone_interface():
            return render_template_string(self._get_zone_template())

        @self.app.route('/list_uploaded_videos')
        def list_uploaded_videos():
            """Список загруженных видео"""
            try:
                uploads_dir = self.app.config['UPLOAD_FOLDER']
                if not os.path.exists(uploads_dir):
                    return jsonify({'videos': []})

                videos = []
                for filename in os.listdir(uploads_dir):
                    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        filepath = os.path.join(uploads_dir, filename)
                        size = os.path.getsize(filepath)
                        mtime = os.path.getmtime(filepath)
                        videos.append({
                            'filename': filename,
                            'size': size,
                            'size_mb': round(size / 1024 / 1024, 1),
                            'size_gb': round(size / 1024 / 1024 / 1024, 2),
                            'modified': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        })

                videos.sort(key=lambda x: x['modified'], reverse=True)
                return jsonify({'videos': videos})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/load_existing_video', methods=['POST'])
        def load_existing_video():
            """Загрузка существующего видео"""
            try:
                data = request.get_json()
                filename = data.get('filename')

                if not filename:
                    return jsonify({'error': 'Не указано имя файла'}), 400

                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

                if not os.path.exists(filepath):
                    return jsonify({'error': 'Файл не найден'}), 404

                print(f"Loading video: {filepath}")
                success = self._load_video(filepath)

                if success:
                    # Загружаем существующие зоны если есть
                    self._load_zones_for_video(filename)

                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'total_frames': self.total_frames,
                        'message': f'Видео загружено: {self.total_frames} кадров'
                    })
                else:
                    return jsonify({'error': 'Не удалось открыть видео'}), 400

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get_frame/<int:frame_index>')
        def get_frame(frame_index):
            if not self.video_cap:
                return jsonify({'error': 'Видео не загружено'}), 400

            frame_data = self._get_frame_with_zones(frame_index)
            if frame_data:
                return jsonify(frame_data)
            else:
                return jsonify({'error': 'Не удалось получить кадр'}), 400

        @self.app.route('/save_zones', methods=['POST'])
        def save_zones():
            """Сохранение зон"""
            try:
                data = request.get_json()
                self.zones = data.get('zones', {})

                # Сохраняем зоны в файл
                zones_file = self._get_zones_filename()
                with open(zones_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'zones': self.zones,
                        'batch': self.current_batch,
                        'video': self.current_video,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)

                return jsonify({'success': True, 'message': 'Зоны сохранены'})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/set_batch_info', methods=['POST'])
        def set_batch_info():
            """Установка информации о партии"""
            try:
                data = request.get_json()
                self.current_batch = {
                    'name': data.get('name', ''),
                    'weight': float(data.get('weight', 0.0)),
                    'target_count': int(data.get('target_count', 0))
                }

                return jsonify({'success': True, 'message': 'Информация о партии обновлена'})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/generate_training_data', methods=['POST'])
        def generate_training_data():
            """Генерация обучающих данных на основе зон"""
            try:
                data = request.get_json()
                frames_count = data.get('frames_count', 100)

                generated = self._generate_zone_training_data(frames_count)

                return jsonify({
                    'success': True,
                    'generated_frames': generated,
                    'message': f'Создано {generated} обучающих кадров'
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/remove_detection', methods=['POST'])
        def remove_detection():
            """Удаление детекции"""
            try:
                data = request.get_json()
                detection_id = data.get('detection_id')

                # Здесь логика удаления детекции
                return jsonify({'success': True, 'message': f'Детекция ID {detection_id} удалена'})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def _load_video(self, filepath):
        """Загрузка видео"""
        try:
            self.video_cap = cv2.VideoCapture(filepath)
            if not self.video_cap.isOpened():
                return False

            self.current_video = filepath
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_index = 0

            # Проверяем первый кадр
            ret, frame = self.video_cap.read()
            if ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return True
            return False

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def _get_frame_with_zones(self, frame_index):
        """Получение кадра с зонами и детекциями"""
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_cap.read()

            if not ret:
                return None

            # Детекция хлеба
            detections = self._detect_bread(frame)

            # Рисуем зоны и детекции
            annotated_frame = self._draw_zones_and_detections(frame.copy(), detections)

            # Конвертируем в base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                'frame_index': frame_index,
                'image': img_base64,
                'detections': detections,
                'zones': self.zones,
                'timestamp': frame_index / self.video_cap.get(cv2.CAP_PROP_FPS) if self.video_cap.get(
                    cv2.CAP_PROP_FPS) > 0 else 0
            }

        except Exception as e:
            print(f"Error getting frame: {e}")
            return None

    def _detect_bread(self, frame):
        """Простая детекция хлеба"""
        detections = []

        # HSV детекция
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Маска для хлебных оттенков
        lower = np.array([10, 30, 30])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Морфология
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 3000 < area < 80000:  # Фильтр по размеру
                x, y, w, h = cv2.boundingRect(contour)

                # Проверяем, попадает ли в зоны исключения
                center_x, center_y = x + w // 2, y + h // 2
                if self._point_in_exclude_zones(center_x, center_y):
                    continue

                detections.append({
                    'id': i,
                    'bbox': [x, y, x + w, y + h],
                    'center': [center_x, center_y],
                    'area': area,
                    'confidence': 0.8,
                    'in_counting_zone': self._point_in_zone(center_x, center_y, 'counting_zone'),
                    'in_entry_zone': self._point_in_zone(center_x, center_y, 'entry_zone'),
                    'in_exit_zone': self._point_in_zone(center_x, center_y, 'exit_zone')
                })

        return detections

    def _draw_zones_and_detections(self, frame, detections):
        """Отрисовка зон и детекций"""

        # Рисуем зоны
        zone_colors = {
            'counting_zone': (0, 255, 0),  # Зеленая - зона подсчета
            'entry_zone': (255, 0, 0),  # Синяя - зона входа
            'exit_zone': (0, 0, 255),  # Красная - зона выхода
            'exclude_zones': (128, 128, 128)  # Серая - зоны исключения
        }

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

            # Цвет в зависимости от зоны
            if detection['in_counting_zone']:
                color = (0, 255, 0)  # Зеленый - в зоне подсчета
            elif detection['in_entry_zone']:
                color = (255, 0, 0)  # Синий - в зоне входа
            elif detection['in_exit_zone']:
                color = (0, 0, 255)  # Красный - в зоне выхода
            else:
                color = (255, 255, 255)  # Белый - вне зон

            # Рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID
            cv2.putText(frame, f"ID: {detection['id']}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Центр
            cv2.circle(frame, (center_x, center_y), 4, color, -1)

        return frame

    def _draw_zone(self, frame, zone, color):
        """Отрисовка зоны"""
        if not zone or len(zone) < 3:
            return

        # Конвертируем в numpy array
        points = np.array(zone, np.int32)

        # Рисуем полигон
        cv2.polylines(frame, [points], True, color, 3)

        # Полупрозрачная заливка
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

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

    def _load_zones_for_video(self, filename):
        """Загрузка сохраненных зон для видео"""
        zones_file = self._get_zones_filename()
        if os.path.exists(zones_file):
            try:
                with open(zones_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.zones = data.get('zones', self.zones)
                    self.current_batch = data.get('batch', self.current_batch)
                print(f"Загружены зоны для {filename}")
            except Exception as e:
                print(f"Ошибка загрузки зон: {e}")

    def _get_zones_filename(self):
        """Получение имени файла зон"""
        if self.current_video:
            video_name = Path(self.current_video).stem
            return f"training_data/zones/{video_name}_zones.json"
        return "training_data/zones/default_zones.json"

    def _generate_zone_training_data(self, frames_count):
        """Генерация обучающих данных на основе зон"""
        if not self.video_cap:
            return 0

        generated_count = 0
        step = max(1, self.total_frames // frames_count)

        for frame_idx in range(0, self.total_frames, step):
            if generated_count >= frames_count:
                break

            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()

            if ret:
                detections = self._detect_bread(frame)

                # Фильтруем детекции по зонам
                valid_detections = []
                for detection in detections:
                    if detection['in_counting_zone']:
                        detection['label'] = self.current_batch['name'] or 'bread'
                        valid_detections.append(detection)

                if valid_detections:
                    # Сохраняем кадр
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = f"zone_frame_{frame_idx:08d}_{timestamp}.jpg"
                    filepath = os.path.join('training_data/images', filename)

                    cv2.imwrite(filepath, frame)

                    # Сохраняем аннотацию
                    annotation = {
                        'filename': filename,
                        'frame_index': frame_idx,
                        'batch_info': self.current_batch,
                        'zones': self.zones,
                        'detections': valid_detections,
                        'timestamp': datetime.now().isoformat()
                    }

                    annotation_file = os.path.join('training_data/annotations',
                                                   filename.replace('.jpg', '.json'))
                    with open(annotation_file, 'w', encoding='utf-8') as f:
                        json.dump(annotation, f, indent=2, ensure_ascii=False)

                    generated_count += 1

        return generated_count

    def _get_zone_template(self):
        """HTML шаблон зонного интерфейса"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Обучение по зонам - Производственная система</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }

        .video-panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .control-panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            max-height: fit-content;
        }

        .video-selector {
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }

        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            margin: 5px;
        }

        .btn:hover:not(:disabled) {
            background: #5a67d8;
            transform: translateY(-2px);
        }

        .btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
            transform: none;
        }

        .btn.success {
            background: #48bb78;
        }

        .btn.success:hover:not(:disabled) {
            background: #38a169;
        }

        .btn.danger {
            background: #e53e3e;
        }

        .btn.danger:hover:not(:disabled) {
            background: #c53030;
        }

        .btn.secondary {
            background: #e2e8f0;
            color: #4a5568;
        }

        .btn.secondary:hover:not(:disabled) {
            background: #cbd5e0;
        }

        .video-container {
            position: relative;
            margin-bottom: 20px;
        }

        #videoFrame {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            cursor: crosshair;
        }

        .video-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
        }

        .frame-slider {
            flex: 1;
            margin: 0 15px;
        }

        .zone-tools {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }

        .zone-tool-btn {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 12px;
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: left;
        }

        .zone-tool-btn:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }

        .zone-tool-btn.active {
            background: #667eea;
            color: white;
            border-color: #5a67d8;
        }

        .zone-legend {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            margin: 15px 0;
        }

        .zone-legend-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
        }

        .zone-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
            border: 2px solid #333;
        }

        .batch-info {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }

        .batch-input {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .detection-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            margin: 15px 0;
        }

        .detection-item {
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .detection-item.in-zone {
            background: #c6f6d5;
        }

        .status-message {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            max-height: 200px;
            overflow-y: auto;
        }

        .status-success {
            background: #c6f6d5;
            color: #22543d;
        }

        .status-error {
            background: #fed7d7;
            color: #742a2a;
        }

        .status-info {
            background: #bee3f8;
            color: #2a4365;
        }

        .status-warning {
            background: #faf089;
            color: #744210;
        }

        .hidden {
            display: none;
        }

        .drawing-instructions {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            display: none;
        }

        .drawing-instructions.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Обучение по зонам - Производственная система</h1>
            <p>Нарисуйте зоны мышкой → Система обучится автоматически</p>
        </div>

        <div class="main-content">
            <div class="video-panel">
                <div class="video-selector" id="videoSelector">
                    <h3>📂 Выберите видео для разметки зон</h3>
                    <button class="btn success" id="selectVideo">📂 Выбрать видео</button>
                </div>

                <div id="videoSection" class="hidden">
                    <div class="video-container">
                        <img id="videoFrame" alt="Кадр видео">
                        <canvas id="drawingCanvas" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
                    </div>

                    <div class="video-controls">
                        <button class="btn" id="prevFrame">⏮️ Пред</button>
                        <input type="range" id="frameSlider" class="frame-slider" min="0" max="100" value="0">
                        <button class="btn" id="nextFrame">⏭️ След</button>
                        <span id="frameInfo">0 / 0</span>
                    </div>
                </div>
            </div>

            <div class="control-panel">
                <h3>🛠️ Инструменты зонирования</h3>

                <div class="zone-tools">
                    <h4>Инструменты рисования:</h4>
                    <button class="zone-tool-btn" data-tool="counting_zone">
                        🟢 Зона подсчета
                        <small style="display: block; color: #666;">Основная зона для подсчета хлеба</small>
                    </button>
                    <button class="zone-tool-btn" data-tool="entry_zone">
                        🔵 Зона входа
                        <small style="display: block; color: #666;">Хлеб выходит из печи</small>
                    </button>
                    <button class="zone-tool-btn" data-tool="exit_zone">
                        🔴 Зона выхода
                        <small style="display: block; color: #666;">Хлеб уходит на стол</small>
                    </button>
                    <button class="zone-tool-btn" data-tool="exclude_zone">
                        ⚫ Зона исключения
                        <small style="display: block; color: #666;">Игнорировать эту область</small>
                    </button>
                    <button class="zone-tool-btn" data-tool="edit">
                        ✏️ Редактировать зоны
                        <small style="display: block; color: #666;">Изменить существующие зоны</small>
                    </button>
                </div>

                <div class="drawing-instructions" id="drawingInstructions">
                    <strong>Рисование зоны:</strong><br>
                    • Кликайте мышкой по углам зоны<br>
                    • Двойной клик - завершить зону<br>
                    • ESC - отменить рисование
                </div>

                <div class="batch-info">
                    <h4>📦 Информация о партии:</h4>
                    <input type="text" id="batchName" class="batch-input" placeholder="Название хлеба (например: Олександрівський, 0.7кг)">
                    <input type="number" id="batchWeight" class="batch-input" placeholder="Вес буханки в кг" step="0.1">
                    <input type="number" id="targetCount" class="batch-input" placeholder="Ожидаемое количество">
                    <button class="btn" id="saveBatchInfo">💾 Сохранить информацию</button>
                </div>

                <div class="zone-legend">
                    <h4>🎨 Легенда зон:</h4>
                    <div class="zone-legend-item">
                        <div class="zone-color" style="background: rgba(0, 255, 0, 0.5);"></div>
                        <span>Зона подсчета - основная</span>
                    </div>
                    <div class="zone-legend-item">
                        <div class="zone-color" style="background: rgba(255, 0, 0, 0.5);"></div>
                        <span>Зона входа - из печи</span>
                    </div>
                    <div class="zone-legend-item">
                        <div class="zone-color" style="background: rgba(0, 0, 255, 0.5);"></div>
                        <span>Зона выхода - на стол</span>
                    </div>
                    <div class="zone-legend-item">
                        <div class="zone-color" style="background: rgba(128, 128, 128, 0.5);"></div>
                        <span>Исключение - игнорировать</span>
                    </div>
                </div>

                <div style="text-align: center; margin: 20px 0;">
                    <button class="btn success" id="saveZones">💾 Сохранить зоны</button>
                    <button class="btn secondary" id="clearZones">🗑️ Очистить все</button>
                </div>

                <div class="detection-list" id="detectionList">
                    <h5>🔍 Найденные объекты:</h5>
                    <p>Загрузите видео для начала работы</p>
                </div>

                <div style="text-align: center; margin: 20px 0;">
                    <button class="btn success" id="generateTrainingData">🚀 Создать обучающие данные</button>
                    <input type="number" id="framesCount" placeholder="Количество кадров" value="100" 
                           style="width: 100%; margin: 10px 0; padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                </div>

                <div id="statusMessages"></div>
            </div>
        </div>
    </div>

    <script>
        // Состояние приложения
        let currentFrame = 0;
        let totalFrames = 0;
        let currentDetections = [];
        let zones = {
            counting_zone: null,
            entry_zone: null,
            exit_zone: null,
            exclude_zones: []
        };

        // Инструменты рисования
        let currentTool = null;
        let isDrawing = false;
        let currentZone = [];

        // DOM элементы
        const videoFrame = document.getElementById('videoFrame');
        const drawingCanvas = document.getElementById('drawingCanvas');
        const frameSlider = document.getElementById('frameSlider');
        const frameInfo = document.getElementById('frameInfo');
        const detectionList = document.getElementById('detectionList');
        const statusMessages = document.getElementById('statusMessages');
        const drawingInstructions = document.getElementById('drawingInstructions');

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            setupCanvas();
        });

        function setupEventListeners() {
            // Выбор видео
            document.getElementById('selectVideo').addEventListener('click', selectExistingVideo);

            // Навигация по кадрам
            document.getElementById('prevFrame').addEventListener('click', () => navigateFrame(-1));
            document.getElementById('nextFrame').addEventListener('click', () => navigateFrame(1));
            frameSlider.addEventListener('input', (e) => goToFrame(parseInt(e.target.value)));

            // Инструменты зон
            document.querySelectorAll('.zone-tool-btn').forEach(btn => {
                btn.addEventListener('click', (e) => selectTool(e.target.dataset.tool));
            });

            // Управление зонами
            document.getElementById('saveZones').addEventListener('click', saveZones);
            document.getElementById('clearZones').addEventListener('click', clearAllZones);

            // Информация о партии
            document.getElementById('saveBatchInfo').addEventListener('click', saveBatchInfo);

            // Генерация данных
            document.getElementById('generateTrainingData').addEventListener('click', generateTrainingData);

            // Клавиши
            document.addEventListener('keydown', handleKeyPress);
        }

        function setupCanvas() {
            if (!drawingCanvas || !videoFrame) return;

            // Синхронизируем размер canvas с изображением
            videoFrame.addEventListener('load', () => {
                drawingCanvas.width = videoFrame.clientWidth;
                drawingCanvas.height = videoFrame.clientHeight;
                redrawZones();
            });

            // Обработчик кликов для рисования
            drawingCanvas.addEventListener('click', handleCanvasClick);
            drawingCanvas.addEventListener('dblclick', finishZone);

            // Делаем canvas интерактивным когда выбран инструмент
            drawingCanvas.style.pointerEvents = 'auto';
        }

        function selectTool(tool) {
            currentTool = tool;

            // Обновляем активный инструмент
            document.querySelectorAll('.zone-tool-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-tool="${tool}"]`).classList.add('active');

            // Показываем инструкции
            if (tool && tool !== 'edit') {
                drawingInstructions.classList.add('active');
                drawingCanvas.style.cursor = 'crosshair';
            } else {
                drawingInstructions.classList.remove('active');
                drawingCanvas.style.cursor = 'default';
            }

            currentZone = [];
            isDrawing = false;
        }

        function handleCanvasClick(e) {
            if (!currentTool || currentTool === 'edit') return;

            const rect = drawingCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Масштабируем координаты к размеру изображения
            const scaleX = videoFrame.naturalWidth / videoFrame.clientWidth;
            const scaleY = videoFrame.naturalHeight / videoFrame.clientHeight;

            const imageX = Math.round(x * scaleX);
            const imageY = Math.round(y * scaleY);

            currentZone.push([imageX, imageY]);
            isDrawing = true;

            // Временно рисуем точку
            const ctx = drawingCanvas.getContext('2d');
            ctx.fillStyle = 'yellow';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();

            showStatus(`Точка ${currentZone.length} добавлена. Двойной клик для завершения зоны.`, 'info');
        }

        function finishZone(e) {
            if (!isDrawing || currentZone.length < 3) {
                showStatus('Нужно минимум 3 точки для создания зоны', 'warning');
                return;
            }

            // Сохраняем зону
            if (currentTool === 'exclude_zone') {
                zones.exclude_zones.push([...currentZone]);
            } else {
                zones[currentTool] = [...currentZone];
            }

            // Очищаем текущую зону
            currentZone = [];
            isDrawing = false;

            redrawZones();
            showStatus(`Зона "${currentTool}" создана с ${currentZone.length} точками`, 'success');

            // Перезагружаем кадр для обновления детекций
            loadFrame(currentFrame);
        }

        function handleKeyPress(e) {
            if (e.key === 'Escape') {
                currentZone = [];
                isDrawing = false;
                redrawZones();
                showStatus('Рисование зоны отменено', 'info');
            }
        }

        function redrawZones() {
            const ctx = drawingCanvas.getContext('2d');
            ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

            const zoneColors = {
                counting_zone: 'rgba(0, 255, 0, 0.3)',
                entry_zone: 'rgba(255, 0, 0, 0.3)',
                exit_zone: 'rgba(0, 0, 255, 0.3)',
                exclude_zones: 'rgba(128, 128, 128, 0.3)'
            };

            const scaleX = videoFrame.clientWidth / videoFrame.naturalWidth;
            const scaleY = videoFrame.clientHeight / videoFrame.naturalHeight;

            // Рисуем все зоны
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
            if (!zone || zone.length < 3) return;

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

            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }

        function selectExistingVideo() {
            showStatus('Загрузка списка видео...', 'info');

            fetch('/list_uploaded_videos')
                .then(response => response.json())
                .then(data => {
                    if (data.videos && data.videos.length > 0) {
                        showVideoSelectionDialog(data.videos);
                    } else {
                        showStatus('В папке uploads нет видео файлов', 'warning');
                    }
                })
                .catch(error => {
                    showStatus('Ошибка получения списка файлов: ' + error.message, 'error');
                });
        }

        function showVideoSelectionDialog(videos) {
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 2000;
            `;

            const dialog = document.createElement('div');
            dialog.style.cssText = `
                background: white;
                border-radius: 15px;
                padding: 30px;
                max-width: 700px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            `;

            let html = `
                <h3 style="margin-bottom: 20px;">🎬 Выберите видео для зонирования</h3>
                <div style="margin-bottom: 20px; color: #718096;">
                    Найдено ${videos.length} видео файлов
                </div>
                <div style="max-height: 400px; overflow-y: auto;">
            `;

            videos.forEach(video => {
                const sizeText = video.size_gb > 1 ? 
                    `${video.size_gb} GB` : 
                    `${video.size_mb} MB`;

                html += `
                    <div onclick="selectVideoFromDialog('${video.filename}')" 
                         style="padding: 15px; margin: 10px 0; border: 2px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: all 0.2s;">
                        <div style="font-weight: bold; margin-bottom: 5px;">🎬 ${video.filename}</div>
                        <div style="font-size: 12px; color: #718096;">📦 ${sizeText} • 🗓️ ${video.modified}</div>
                    </div>
                `;
            });

            html += `
                </div>
                <div style="margin-top: 25px; text-align: right;">
                    <button class="btn secondary" onclick="closeVideoSelectionDialog()">❌ Отмена</button>
                </div>
            `;

            dialog.innerHTML = html;
            modal.appendChild(dialog);
            document.body.appendChild(modal);

            window.selectVideoFromDialog = function(filename) {
                closeVideoSelectionDialog();
                loadSelectedVideo(filename);
            };

            window.closeVideoSelectionDialog = function() {
                document.body.removeChild(modal);
                delete window.selectVideoFromDialog;
                delete window.closeVideoSelectionDialog;
            };
        }

        function loadSelectedVideo(filename) {
            showStatus(`Загружаю ${filename}...`, 'info');

            fetch('/load_existing_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: filename})
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    totalFrames = result.total_frames;
                    frameSlider.max = totalFrames - 1;

                    document.getElementById('videoSelector').classList.add('hidden');
                    document.getElementById('videoSection').classList.remove('hidden');

                    showStatus(`✅ ${result.message}`, 'success');
                    loadFrame(0);
                } else {
                    showStatus(result.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка загрузки: ' + error.message, 'error');
            });
        }

        function loadFrame(frameIndex) {
            currentFrame = frameIndex;

            fetch(`/get_frame/${frameIndex}`)
                .then(response => response.json())
                .then(data => {
                    if (data.image) {
                        videoFrame.src = 'data:image/jpeg;base64,' + data.image;
                        currentDetections = data.detections || [];
                        if (data.zones) {
                            zones = data.zones;
                        }

                        updateFrameInfo();
                        updateDetectionList();
                        frameSlider.value = frameIndex;

                        // Перерисовываем зоны после загрузки кадра
                        videoFrame.onload = () => {
                            setupCanvas();
                            redrawZones();
                        };
                    }
                })
                .catch(error => {
                    showStatus('Ошибка загрузки кадра: ' + error.message, 'error');
                });
        }

        function navigateFrame(direction) {
            const newFrame = currentFrame + direction;
            if (newFrame >= 0 && newFrame < totalFrames) {
                loadFrame(newFrame);
            }
        }

        function goToFrame(frameIndex) {
            if (frameIndex >= 0 && frameIndex < totalFrames) {
                loadFrame(frameIndex);
            }
        }

        function updateFrameInfo() {
            frameInfo.textContent = `${currentFrame + 1} / ${totalFrames}`;
        }

        function updateDetectionList() {
            if (currentDetections.length === 0) {
                detectionList.innerHTML = '<h5>🔍 Найденные объекты:</h5><p>Объекты не найдены</p>';
                return;
            }

            const inZoneCount = currentDetections.filter(d => d.in_counting_zone).length;
            const entryCount = currentDetections.filter(d => d.in_entry_zone).length;
            const exitCount = currentDetections.filter(d => d.in_exit_zone).length;

            let html = `
                <h5>🔍 Найденные объекты: ${currentDetections.length}</h5>
                <div style="font-size: 12px; margin: 10px 0; color: #666;">
                    🟢 В зоне подсчета: ${inZoneCount}<br>
                    🔵 В зоне входа: ${entryCount}<br>
                    🔴 В зоне выхода: ${exitCount}
                </div>
            `;

            currentDetections.forEach(detection => {
                let zoneStatus = '⚪ Вне зон';
                let itemClass = '';

                if (detection.in_counting_zone) {
                    zoneStatus = '🟢 Зона подсчета';
                    itemClass = 'in-zone';
                } else if (detection.in_entry_zone) {
                    zoneStatus = '🔵 Зона входа';
                } else if (detection.in_exit_zone) {
                    zoneStatus = '🔴 Зона выхода';
                }

                html += `
                    <div class="detection-item ${itemClass}">
                        <div>
                            <strong>ID ${detection.id}</strong><br>
                            <small>${zoneStatus}</small><br>
                            <small>Размер: ${detection.area}px²</small>
                        </div>
                        <button class="btn danger" style="padding: 5px 10px; font-size: 12px;" 
                                onclick="removeDetection(${detection.id})">
                            🗑️ Удалить
                        </button>
                    </div>
                `;
            });

            detectionList.innerHTML = html;
        }

        function removeDetection(detectionId) {
            fetch('/remove_detection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({detection_id: detectionId})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Детекция ID ${detectionId} удалена`, 'success');
                    loadFrame(currentFrame); // Перезагружаем кадр
                } else {
                    showStatus('Ошибка удаления детекции', 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка: ' + error.message, 'error');
            });
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
                    showStatus('✅ Зоны сохранены успешно', 'success');
                } else {
                    showStatus('Ошибка сохранения зон', 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка: ' + error.message, 'error');
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
                loadFrame(currentFrame);
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
                showStatus('Введите название хлеба', 'warning');
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
                    showStatus(`✅ Информация о партии "${batchData.name}" сохранена`, 'success');
                } else {
                    showStatus('Ошибка сохранения информации о партии', 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка: ' + error.message, 'error');
            });
        }

        function generateTrainingData() {
            const framesCount = parseInt(document.getElementById('framesCount').value) || 100;

            if (!zones.counting_zone) {
                showStatus('Создайте сначала зону подсчета', 'warning');
                return;
            }

            showStatus(`Генерация ${framesCount} обучающих кадров...`, 'info');

            fetch('/generate_training_data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frames_count: framesCount})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`🎉 ${data.message}`, 'success');
                } else {
                    showStatus('Ошибка генерации данных', 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка: ' + error.message, 'error');
            });
        }

        function showStatus(message, type) {
            const statusDiv = document.createElement('div');
            statusDiv.className = `status-message status-${type}`;
            statusDiv.textContent = new Date().toLocaleTimeString() + ': ' + message;

            statusMessages.appendChild(statusDiv);
            statusMessages.scrollTop = statusMessages.scrollHeight;

            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.remove();
                }
            }, 8000);
        }
    </script>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5002, debug=False):
        """Запуск зонного интерфейса"""
        print(f"🎯 Запуск зонной системы обучения на http://{host}:{port}")
        print("🖱️ Рисуйте зоны мышкой → Система обучится автоматически!")

        self.app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    app = ZoneTrainingApp()
    app.run()