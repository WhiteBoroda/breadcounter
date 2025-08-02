# interactive_training_web.py - Веб-интерфейс для интерактивного обучения
from flask import Flask, render_template_string, request, jsonify, send_file, url_for
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


class InteractiveTrainingApp:
    """Веб-приложение для интерактивного обучения системы"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Настройки загрузки
        self.app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
        self.app.config['UPLOAD_FOLDER'] = 'uploads'

        # Создаем папки
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('training_data/images', exist_ok=True)
        os.makedirs('training_data/annotations', exist_ok=True)

        # Текущее состояние
        self.current_video = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.video_cap = None

        # Данные обучения
        self.training_data = []
        self.bread_types = ['white_bread', 'dark_bread', 'baton', 'molded_bread']

        self._setup_routes()

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.route('/')
        def training_interface():
            return render_template_string(self._get_training_template())

        @self.app.route('/upload_video', methods=['POST'])
        def upload_video():
            try:
                print("=== UPLOAD REQUEST RECEIVED ===")
                print(f"Request method: {request.method}")
                print(f"Content length: {request.content_length}")
                print(f"Request files: {list(request.files.keys())}")

                if 'video' not in request.files:
                    print("ERROR: No video file in request")
                    return jsonify({'error': 'Нет файла'}), 400

                file = request.files['video']
                print(f"File received: {file.filename}")
                print(f"File content type: {file.content_type}")

                if file.filename == '':
                    print("ERROR: Empty filename")
                    return jsonify({'error': 'Файл не выбран'}), 400

                if file and self._allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

                    print(f"Saving file to: {filepath}")
                    print("Starting file save...")

                    # Сохраняем файл частями для больших файлов
                    chunk_size = 8192  # 8KB chunks
                    with open(filepath, 'wb') as f:
                        while True:
                            chunk = file.stream.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)

                    print("File saved successfully")

                    # Проверяем что файл сохранился
                    if not os.path.exists(filepath):
                        print("ERROR: File was not saved")
                        return jsonify({'error': 'Ошибка сохранения файла'}), 500

                    file_size = os.path.getsize(filepath)
                    print(f"Saved file size: {file_size} bytes")

                    # Инициализируем видео
                    print("Loading video...")
                    success = self._load_video(filepath)
                    if success:
                        print(f"Video loaded successfully: {self.total_frames} frames")
                        return jsonify({
                            'success': True,
                            'filename': filename,
                            'total_frames': self.total_frames,
                            'message': f'Видео загружено: {self.total_frames} кадров'
                        })
                    else:
                        print("ERROR: Failed to load video")
                        return jsonify({'error': 'Не удалось открыть видео'}), 400

                else:
                    print("ERROR: Invalid file type")
                    return jsonify({'error': 'Неподдерживаемый формат файла'}), 400

            except Exception as e:
                print(f"CRITICAL ERROR in upload_video: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                return jsonify({'error': f'Ошибка сервера: {str(e)}'}), 500

        @self.app.route('/get_frame/<int:frame_index>')
        def get_frame(frame_index):
            if not self.video_cap:
                return jsonify({'error': 'Видео не загружено'}), 400

            frame_data = self._get_frame_with_detections(frame_index)
            if frame_data:
                return jsonify(frame_data)
            else:
                return jsonify({'error': 'Не удалось получить кадр'}), 400

        @self.app.route('/annotate_frame', methods=['POST'])
        def annotate_frame():
            data = request.get_json()
            frame_index = data.get('frame_index')
            annotations = data.get('annotations', [])
            bread_type = data.get('bread_type', 'unknown')

            success = self._save_annotation(frame_index, annotations, bread_type)

            if success:
                return jsonify({'success': True, 'message': 'Аннотация сохранена'})
            else:
                return jsonify({'error': 'Ошибка сохранения'}), 400

        @self.app.route('/get_training_progress')
        def get_training_progress():
            return jsonify({
                'total_annotations': len(self.training_data),
                'bread_types_count': self._count_bread_types(),
                'ready_for_training': len(self.training_data) >= 100
            })

        @self.app.route('/start_training', methods=['POST'])
        def start_training():
            if len(self.training_data) < 50:
                return jsonify({'error': 'Недостаточно данных для обучения'}), 400

            # Запускаем обучение в фоне
            success = self._start_model_training()

            if success:
                return jsonify({'success': True, 'message': 'Обучение запущено'})
            else:
                return jsonify({'error': 'Ошибка запуска обучения'}), 400

        @self.app.route('/load_existing_video', methods=['POST'])
        def load_existing_video():
            """Загрузка существующего видео из папки uploads"""
            try:
                data = request.get_json()
                filename = data.get('filename')

                if not filename:
                    return jsonify({'error': 'Не указано имя файла'}), 400

                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

                if not os.path.exists(filepath):
                    return jsonify({'error': 'Файл не найден'}), 404

                print(f"Loading existing video: {filepath}")
                success = self._load_video(filepath)

                if success:
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'total_frames': self.total_frames,
                        'message': f'Видео загружено: {self.total_frames} кадров'
                    })
                else:
                    return jsonify({'error': 'Не удалось открыть видео'}), 400

            except Exception as e:
                print(f"Error loading existing video: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/auto_extract_frames', methods=['POST'])
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
                        videos.append({
                            'filename': filename,
                            'size': size,
                            'size_mb': round(size / 1024 / 1024, 1)
                        })

                return jsonify({'videos': videos})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        def auto_extract_frames():
            data = request.get_json()
            interval = data.get('interval', 30)
            max_frames = data.get('max_frames', 200)

            extracted = self._auto_extract_frames(interval, max_frames)

            return jsonify({
                'success': True,
                'extracted_frames': extracted,
                'message': f'Извлечено {extracted} кадров'
            })

    def _allowed_file(self, filename):
        """Проверка допустимых форматов"""
        allowed = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

    def _load_video(self, filepath):
        """Загрузка видео"""
        try:
            print(f"Loading video: {filepath}")
            print(f"File exists: {os.path.exists(filepath)}")
            print(f"File size: {os.path.getsize(filepath) if os.path.exists(filepath) else 'N/A'} bytes")

            self.video_cap = cv2.VideoCapture(filepath)
            if not self.video_cap.isOpened():
                print("Failed to open video with OpenCV")
                return False

            self.current_video = filepath
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_index = 0

            print(f"Video loaded successfully: {self.total_frames} frames")

            # Проверяем что можем прочитать первый кадр
            ret, frame = self.video_cap.read()
            if ret:
                print(f"First frame read successfully: {frame.shape}")
                # Возвращаемся к началу
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return True
            else:
                print("Failed to read first frame")
                return False

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def _get_frame_with_detections(self, frame_index):
        """Получение кадра с детекциями"""
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_cap.read()

            if not ret:
                return None

            # Простая детекция для демонстрации
            detections = self._simple_bread_detection(frame)

            # Рисуем детекции на кадре
            annotated_frame = self._draw_detections(frame.copy(), detections)

            # Конвертируем в base64 для передачи
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                'frame_index': frame_index,
                'image': img_base64,
                'detections': detections,
                'timestamp': frame_index / self.video_cap.get(cv2.CAP_PROP_FPS) if self.video_cap.get(
                    cv2.CAP_PROP_FPS) > 0 else 0
            }

        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
            return None

    def _simple_bread_detection(self, frame):
        """Простая детекция хлеба для демонстрации"""
        detections = []

        # Конвертируем в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Маска для хлебных цветов
        lower_bread = np.array([10, 30, 30])
        upper_bread = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_bread, upper_bread)

        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if 2000 < area < 50000:  # Фильтр по размеру
                x, y, w, h = cv2.boundingRect(contour)

                detection = {
                    'id': i,
                    'bbox': [x, y, x + w, y + h],
                    'center': [x + w // 2, y + h // 2],
                    'area': area,
                    'confidence': 0.8,
                    'type': 'unknown'  # Будет определяться пользователем
                }
                detections.append(detection)

        return detections

    def _draw_detections(self, frame, detections):
        """Отрисовка детекций на кадре"""
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ID
            cv2.putText(frame, f"ID: {detection['id']}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Тип (если определен)
            if detection['type'] != 'unknown':
                cv2.putText(frame, detection['type'],
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame

    def _save_annotation(self, frame_index, annotations, bread_type):
        """Сохранение аннотации"""
        try:
            annotation_data = {
                'frame_index': frame_index,
                'video_file': self.current_video,
                'timestamp': datetime.now().isoformat(),
                'bread_type': bread_type,
                'annotations': annotations
            }

            self.training_data.append(annotation_data)

            # Сохраняем в файл
            annotation_file = f"training_data/annotations/frame_{frame_index:06d}.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)

            # Сохраняем кадр
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_cap.read()
            if ret:
                frame_file = f"training_data/images/frame_{frame_index:06d}.jpg"
                cv2.imwrite(frame_file, frame)

            return True

        except Exception as e:
            print(f"Ошибка сохранения аннотации: {e}")
            return False

    def _count_bread_types(self):
        """Подсчет типов хлеба в аннотациях"""
        counts = {}
        for data in self.training_data:
            bread_type = data['bread_type']
            counts[bread_type] = counts.get(bread_type, 0) + 1
        return counts

    def _auto_extract_frames(self, interval, max_frames):
        """Автоматическое извлечение кадров"""
        if not self.video_cap:
            return 0

        extracted = 0
        frame_index = 0

        while extracted < max_frames and frame_index < self.total_frames:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_cap.read()

            if ret:
                # Простая детекция
                detections = self._simple_bread_detection(frame)

                if len(detections) > 0:  # Сохраняем только кадры с детекциями
                    # Сохраняем кадр
                    frame_file = f"training_data/images/auto_frame_{frame_index:06d}.jpg"
                    cv2.imwrite(frame_file, frame)

                    # Создаем базовую аннотацию
                    annotation_data = {
                        'frame_index': frame_index,
                        'video_file': self.current_video,
                        'timestamp': datetime.now().isoformat(),
                        'bread_type': 'needs_labeling',
                        'annotations': detections,
                        'auto_extracted': True
                    }

                    annotation_file = f"training_data/annotations/auto_frame_{frame_index:06d}.json"
                    with open(annotation_file, 'w', encoding='utf-8') as f:
                        json.dump(annotation_data, f, indent=2, ensure_ascii=False)

                    extracted += 1

            frame_index += interval

        return extracted

    def _start_model_training(self):
        """Запуск обучения модели"""
        try:
            # Здесь будет интеграция с системой обучения
            print("Запуск обучения модели...")

            # Создаем файл конфигурации обучения
            training_config = {
                'total_samples': len(self.training_data),
                'bread_types': list(self._count_bread_types().keys()),
                'training_started': datetime.now().isoformat()
            }

            with open('training_data/training_config.json', 'w') as f:
                json.dump(training_config, f, indent=2)

            return True

        except Exception as e:
            print(f"Ошибка запуска обучения: {e}")
            return False

    def _get_training_template(self):
        """HTML шаблон для интерфейса обучения"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Интерактивное обучение системы</title>
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
            max-width: 1400px;
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
            grid-template-columns: 1fr 400px;
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
        }

        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #fafafa;
            position: relative;
            overflow: hidden;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
            transform: translateY(-2px);
        }

        .upload-area.drag-over {
            border-color: #667eea;
            background: #f0f4ff;
            border-style: solid;
        }

        .video-container {
            position: relative;
            margin-bottom: 20px;
        }

        #videoFrame {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .video-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
        }

        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .btn:hover {
            background: #5a67d8;
            transform: translateY(-2px);
        }

        .btn.secondary {
            background: #e2e8f0;
            color: #4a5568;
        }

        .btn.secondary:hover {
            background: #cbd5e0;
        }

        .frame-slider {
            flex: 1;
            margin: 0 15px;
        }

        .bread-types {
            margin: 20px 0;
        }

        .bread-type-btn {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 15px;
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .bread-type-btn:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }

        .bread-type-btn.active {
            background: #667eea;
            color: white;
            border-color: #5a67d8;
        }

        .progress-section {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .progress-bar {
            width: 100%;
            height: 10px;
            background: #e2e8f0;
            border-radius: 5px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: #48bb78;
            transition: width 0.3s ease;
        }

        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }

        .stat-item {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 8px;
        }

        .detection-list {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            margin: 15px 0;
        }

        .detection-item {
            padding: 8px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            cursor: pointer;
        }

        .detection-item:hover {
            background: #e2e8f0;
        }

        .status-message {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
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

        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Интерактивное обучение системы</h1>
            <p>Загрузите видео и разметьте типы хлеба для обучения ИИ</p>
        </div>

        <div class="main-content">
            <div class="video-panel">
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="videoInput" accept="video/*,.mp4,.avi,.mov,.mkv" style="
                        position: absolute;
                        width: 100%;
                        height: 100%;
                        opacity: 0;
                        cursor: pointer;
                        z-index: 2;
                    ">
                    <div style="position: relative; z-index: 1; pointer-events: none;">
                        <h3>📁 Загрузите видео записи</h3>
                        <p><strong>Кликните здесь</strong> для выбора MP4 файла</p>
                        <p>или <strong>перетащите файл</strong> в эту область</p>
                        <small style="color: #666;">Поддерживаются: MP4, AVI, MOV, MKV (до 2GB)</small>
                        <br><small style="color: #999;">Большие файлы могут загружаться 1-5 минут</small>
                    </div>
                </div>

                <div id="videoSection" class="hidden">
                    <div class="video-container">
                        <img id="videoFrame" alt="Кадр видео">
                    </div>

                    <div class="video-controls">
                        <button class="btn" id="prevFrame">⏮️ Пред</button>
                        <input type="range" id="frameSlider" class="frame-slider" min="0" max="100" value="0">
                        <button class="btn" id="nextFrame">⏭️ След</button>
                        <span id="frameInfo">0 / 0</span>
                    </div>

                    <div style="margin-top: 15px;">
                        <button class="btn secondary" id="autoExtract">🚀 Авто-извлечение</button>
                        <button class="btn" id="saveAnnotation">💾 Сохранить разметку</button>
                    </div>
                </div>
            </div>

            <div class="control-panel">
                <h3>🎯 Управление обучением</h3>

                <div class="bread-types">
                    <h4>Тип хлеба:</h4>
                    <button class="bread-type-btn" data-type="white_bread">🍞 Белый хлеб</button>
                    <button class="bread-type-btn" data-type="dark_bread">🍞 Черный хлеб</button>
                    <button class="bread-type-btn" data-type="baton">🥖 Батон</button>
                    <button class="bread-type-btn" data-type="molded_bread">📦 Хлеб в формах</button>
                </div>

                <div class="detection-list" id="detectionList">
                    <p>Детекции появятся здесь</p>
                </div>

                <div class="progress-section">
                    <h4>📊 Прогресс обучения</h4>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>

                    <div class="stats">
                        <div class="stat-item">
                            <div id="totalAnnotations">0</div>
                            <small>Аннотаций</small>
                        </div>
                        <div class="stat-item">
                            <div id="readyStatus">Не готово</div>
                            <small>Статус</small>
                        </div>
                    </div>

                    <button class="btn" id="startTraining" disabled>🎓 Начать обучение</button>
                    <button class="btn secondary" id="testServer" style="margin-top: 10px;">🔧 Тест сервера</button>
                    <button class="btn secondary" id="loadExisting" style="margin-top: 10px;">📁 Загрузить существующий файл</button>
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
        let selectedBreadType = 'white_bread';

        // DOM элементы
        const uploadArea = document.getElementById('uploadArea');
        const videoInput = document.getElementById('videoInput');
        const videoSection = document.getElementById('videoSection');
        const videoFrame = document.getElementById('videoFrame');
        const frameSlider = document.getElementById('frameSlider');
        const frameInfo = document.getElementById('frameInfo');
        const detectionList = document.getElementById('detectionList');
        const statusMessages = document.getElementById('statusMessages');

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            updateProgress();
            checkServerConnection();
        });

        function loadExistingVideo() {
            showStatus('Загрузка списка файлов...', 'info');

            fetch('/list_uploaded_videos')
                .then(response => response.json())
                .then(data => {
                    if (data.videos && data.videos.length > 0) {
                        // Показываем диалог выбора файла
                        let videoList = 'Выберите видео:\n\n';
                        data.videos.forEach((video, index) => {
                            videoList += `${index + 1}. ${video.filename} (${video.size_mb}MB)\n`;
                        });

                        const choice = prompt(videoList + '\nВведите номер файла:');
                        const videoIndex = parseInt(choice) - 1;

                        if (videoIndex >= 0 && videoIndex < data.videos.length) {
                            const selectedVideo = data.videos[videoIndex];

                            showStatus(`Загружаю ${selectedVideo.filename}...`, 'info');

                            fetch('/load_existing_video', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({filename: selectedVideo.filename})
                            })
                            .then(response => response.json())
                            .then(result => {
                                if (result.success) {
                                    totalFrames = result.total_frames;
                                    frameSlider.max = totalFrames - 1;

                                    uploadArea.classList.add('hidden');
                                    videoSection.classList.remove('hidden');

                                    showStatus(result.message, 'success');
                                    loadFrame(0);
                                } else {
                                    showStatus(result.error, 'error');
                                }
                            })
                            .catch(error => {
                                showStatus('Ошибка загрузки: ' + error.message, 'error');
                            });
                        }
                    } else {
                        showStatus('В папке uploads нет видео файлов', 'info');
                    }
                })
                .catch(error => {
                    showStatus('Ошибка получения списка файлов: ' + error.message, 'error');
                });
        }

        function testServerConnection() {
            showStatus('Тестирование подключения к серверу...', 'info');

            fetch('/get_training_progress')
                .then(response => {
                    console.log('Test response status:', response.status);
                    if (response.ok) {
                        return response.json();
                    } else {
                        throw new Error(`HTTP ${response.status}`);
                    }
                })
                .then(data => {
                    console.log('Test response data:', data);
                    showStatus('✅ Сервер работает корректно!', 'success');
                })
                .catch(error => {
                    console.error('Test failed:', error);
                    showStatus('❌ Ошибка подключения: ' + error.message, 'error');
                });
        }

        function checkServerConnection() {
            fetch('/get_training_progress')
                .then(response => response.json())
                .then(data => {
                    console.log('Server connection OK');
                    showStatus('Сервер готов к работе', 'success');
                })
                .catch(error => {
                    console.error('Server connection failed:', error);
                    showStatus('Ошибка подключения к серверу', 'error');
                });
        }

        function setupEventListeners() {
            // Обработчики файлового input'а
            videoInput.addEventListener('change', handleFileSelect);

            // Drag and drop обработчики
            uploadArea.addEventListener('dragenter', handleDragEnter);
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('dragleave', handleDragLeave);
            uploadArea.addEventListener('drop', handleDrop);

            // Предотвращаем открытие файла в браузере для всего документа
            document.addEventListener('dragover', (e) => e.preventDefault());
            document.addEventListener('drop', (e) => e.preventDefault());

            // Навигация по кадрам
            document.getElementById('prevFrame').addEventListener('click', () => navigateFrame(-1));
            document.getElementById('nextFrame').addEventListener('click', () => navigateFrame(1));
            frameSlider.addEventListener('input', (e) => goToFrame(parseInt(e.target.value)));

            // Выбор типа хлеба
            document.querySelectorAll('.bread-type-btn').forEach(btn => {
                btn.addEventListener('click', (e) => selectBreadType(e.target.dataset.type));
            });

            // Кнопки действий
            document.getElementById('saveAnnotation').addEventListener('click', saveCurrentAnnotation);
            document.getElementById('autoExtract').addEventListener('click', autoExtractFrames);
            document.getElementById('startTraining').addEventListener('click', startTraining);
            document.getElementById('testServer').addEventListener('click', testServerConnection);
            document.getElementById('loadExisting').addEventListener('click', loadExistingVideo);
        }

        function handleDragEnter(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.add('drag-over');
        }

        function handleDragOver(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        function handleDragLeave(e) {
            e.preventDefault();
            e.stopPropagation();
            // Проверяем что мы действительно покинули область
            if (!uploadArea.contains(e.relatedTarget)) {
                uploadArea.classList.remove('drag-over');
            }
        }

        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();

            console.log('File dropped');
            uploadArea.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            console.log('Files:', files.length);

            if (files.length > 0) {
                const file = files[0];
                console.log('File type:', file.type);
                // Проверяем что это видео файл
                if (file.type.startsWith('video/')) {
                    uploadVideo(file);
                } else {
                    showStatus('Пожалуйста, выберите видео файл (MP4, AVI, MOV и т.д.)', 'error');
                }
            }
        }

        function handleFileSelect(e) {
            console.log('File selected');
            const file = e.target.files[0];
            if (file) {
                console.log('File details:', file.name, file.type, file.size);
                uploadVideo(file);
            }
        }

        function uploadVideo(file) {
            showStatus('Загрузка видео...', 'info');

            const formData = new FormData();
            formData.append('video', file);

            fetch('/upload_video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    totalFrames = data.total_frames;
                    frameSlider.max = totalFrames - 1;

                    uploadArea.classList.add('hidden');
                    videoSection.classList.remove('hidden');

                    showStatus(data.message, 'success');
                    loadFrame(0);
                } else {
                    showStatus(data.error, 'error');
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

                        updateFrameInfo();
                        updateDetectionList();
                        frameSlider.value = frameIndex;
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
                detectionList.innerHTML = '<p>Детекции не найдены на этом кадре</p>';
                return;
            }

            let html = '<h5>Найденные объекты:</h5>';
            currentDetections.forEach((detection, index) => {
                html += `
                    <div class="detection-item" onclick="highlightDetection(${index})">
                        <strong>ID ${detection.id}</strong><br>
                        Размер: ${detection.area}px²<br>
                        Уверенность: ${(detection.confidence * 100).toFixed(1)}%<br>
                        Тип: ${detection.type === 'unknown' ? 'Не определен' : detection.type}
                    </div>
                `;
            });
            detectionList.innerHTML = html;
        }

        function selectBreadType(type) {
            selectedBreadType = type;

            document.querySelectorAll('.bread-type-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            document.querySelector(`[data-type="${type}"]`).classList.add('active');
        }

        function saveCurrentAnnotation() {
            if (currentDetections.length === 0) {
                showStatus('Нет детекций для сохранения', 'error');
                return;
            }

            const annotationData = {
                frame_index: currentFrame,
                annotations: currentDetections,
                bread_type: selectedBreadType
            };

            fetch('/annotate_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(annotationData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Аннотация сохранена!', 'success');
                    updateProgress();
                    navigateFrame(1); // Переходим к следующему кадру
                } else {
                    showStatus(data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка сохранения: ' + error.message, 'error');
            });
        }

        function autoExtractFrames() {
            showStatus('Автоматическое извлечение кадров...', 'info');

            const extractData = {
                interval: 30,
                max_frames: 200
            };

            fetch('/auto_extract_frames', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(extractData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(data.message, 'success');
                    updateProgress();
                } else {
                    showStatus('Ошибка извлечения', 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка: ' + error.message, 'error');
            });
        }

        function startTraining() {
            showStatus('Запуск обучения модели...', 'info');

            fetch('/start_training', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Обучение запущено! Это может занять 30-60 минут.', 'success');
                } else {
                    showStatus(data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка запуска обучения: ' + error.message, 'error');
            });
        }

        function updateProgress() {
            fetch('/get_training_progress')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalAnnotations').textContent = data.total_annotations;

                    const progress = Math.min(data.total_annotations / 100 * 100, 100);
                    document.getElementById('progressFill').style.width = progress + '%';

                    const readyStatus = document.getElementById('readyStatus');
                    const startButton = document.getElementById('startTraining');

                    if (data.ready_for_training) {
                        readyStatus.textContent = 'Готово';
                        readyStatus.style.color = '#22543d';
                        startButton.disabled = false;
                    } else {
                        readyStatus.textContent = 'Нужно больше данных';
                        readyStatus.style.color = '#742a2a';
                        startButton.disabled = true;
                    }
                })
                .catch(error => {
                    console.error('Ошибка обновления прогресса:', error);
                });
        }

        function showStatus(message, type) {
            const statusDiv = document.createElement('div');
            statusDiv.className = `status-message status-${type}`;
            statusDiv.textContent = message;

            statusMessages.appendChild(statusDiv);

            // Автоматическое удаление через 5 секунд
            setTimeout(() => {
                statusDiv.remove();
            }, 5000);
        }

        function highlightDetection(index) {
            // Здесь можно добавить подсветку выбранной детекции на изображении
            console.log('Выбрана детекция:', currentDetections[index]);
        }

        // Обновляем прогресс каждые 30 секунд
        setInterval(updateProgress, 30000);
    </script>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Запуск веб-приложения"""
        print(f"🚀 Запуск интерактивного обучения на http://{host}:{port}")
        print("📁 Загрузите MP4 видео и начните разметку!")

        # Отключаем debug для стабильности с большими файлами
        if debug:
            print("⚠️  Debug режим отключен для обработки больших файлов")
            debug = False

        self.app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    app = InteractiveTrainingApp()
    app.run()