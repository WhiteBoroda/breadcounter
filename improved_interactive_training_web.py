# improved_interactive_training_web.py - Улучшенный веб-интерфейс с загрузкой больших файлов
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
import threading
import uuid


class ImprovedTrainingApp:
    """Улучшенный веб-интерфейс с поддержкой больших файлов"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Увеличенные лимиты для больших файлов
        self.app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        # Временные файлы для чанковой загрузки
        self.app.config['TEMP_FOLDER'] = 'temp_uploads'

        # Создаем папки
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('temp_uploads', exist_ok=True)
        os.makedirs('training_data/images', exist_ok=True)
        os.makedirs('training_data/annotations', exist_ok=True)

        # Состояние загрузки
        self.upload_sessions = {}  # Для отслеживания прогресса

        # Текущее состояние
        self.current_video = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.video_cap = None

        # Данные        обучения
        self.training_data = []
        self.bread_types = ['white_bread', 'dark_bread', 'baton', 'molded_bread', 'defective_bread']

        self._setup_routes()

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.route('/')
        def training_interface():
            return render_template_string(self._get_improved_template())

        @self.app.route('/start_chunked_upload', methods=['POST'])
        def start_chunked_upload():
            """Начало чанковой загрузки"""
            try:
                data = request.get_json()
                filename = secure_filename(data.get('filename', 'video.mp4'))
                file_size = data.get('file_size', 0)

                # Создаем уникальный ID сессии
                session_id = str(uuid.uuid4())

                # Добавляем timestamp к имени файла
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"

                temp_path = os.path.join(self.app.config['TEMP_FOLDER'], f"{session_id}_{filename}")

                self.upload_sessions[session_id] = {
                    'filename': filename,
                    'temp_path': temp_path,
                    'file_size': file_size,
                    'uploaded_size': 0,
                    'start_time': time.time(),
                    'status': 'uploading'
                }

                print(f"Started chunked upload: {filename} ({file_size} bytes)")

                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Загрузка инициализирована'
                })

            except Exception as e:
                print(f"Error starting chunked upload: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/upload_chunk', methods=['POST'])
        def upload_chunk():
            """Загрузка чанка файла"""
            try:
                session_id = request.form.get('session_id')
                chunk_index = int(request.form.get('chunk_index', 0))

                if session_id not in self.upload_sessions:
                    return jsonify({'error': 'Неверная сессия'}), 400

                session = self.upload_sessions[session_id]

                if 'chunk' not in request.files:
                    return jsonify({'error': 'Нет данных чанка'}), 400

                chunk_file = request.files['chunk']
                chunk_data = chunk_file.read()

                # Дописываем чанк к временному файлу
                with open(session['temp_path'], 'ab') as f:
                    f.write(chunk_data)

                session['uploaded_size'] += len(chunk_data)

                progress = (session['uploaded_size'] / session['file_size']) * 100 if session['file_size'] > 0 else 0

                print(f"Chunk {chunk_index} uploaded: {len(chunk_data)} bytes, progress: {progress:.1f}%")

                return jsonify({
                    'success': True,
                    'progress': progress,
                    'uploaded_size': session['uploaded_size']
                })

            except Exception as e:
                print(f"Error uploading chunk: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/finish_upload', methods=['POST'])
        def finish_upload():
            """Завершение загрузки"""
            try:
                data = request.get_json()
                session_id = data.get('session_id')

                if session_id not in self.upload_sessions:
                    return jsonify({'error': 'Неверная сессия'}), 400

                session = self.upload_sessions[session_id]

                # Перемещаем файл в финальную папку
                final_path = os.path.join(self.app.config['UPLOAD_FOLDER'], session['filename'])
                os.rename(session['temp_path'], final_path)

                print(f"Upload finished: {final_path}")

                # Загружаем видео
                success = self._load_video(final_path)

                if success:
                    session['status'] = 'completed'

                    # Очищаем сессию через некоторое время
                    threading.Timer(300, lambda: self.upload_sessions.pop(session_id, None)).start()

                    return jsonify({
                        'success': True,
                        'filename': session['filename'],
                        'total_frames': self.total_frames,
                        'message': f'Видео загружено: {self.total_frames} кадров'
                    })
                else:
                    session['status'] = 'error'
                    return jsonify({'error': 'Не удалось открыть видео'}), 400

            except Exception as e:
                print(f"Error finishing upload: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/upload_progress/<session_id>')
        def upload_progress(session_id):
            """Получение прогресса загрузки"""
            if session_id in self.upload_sessions:
                session = self.upload_sessions[session_id]
                progress = (session['uploaded_size'] / session['file_size']) * 100 if session['file_size'] > 0 else 0

                return jsonify({
                    'progress': progress,
                    'uploaded_size': session['uploaded_size'],
                    'file_size': session['file_size'],
                    'status': session['status']
                })
            else:
                return jsonify({'error': 'Сессия не найдена'}), 404

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

                # Сортируем по дате изменения (новые сначала)
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

        # ... остальные маршруты (get_frame, annotate_frame, etc.) остаются без изменений
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

            success = self._start_model_training()

            if success:
                return jsonify({'success': True, 'message': 'Обучение запущено'})
            else:
                return jsonify({'error': 'Ошибка запуска обучения'}), 400

        @self.app.route('/auto_extract_frames', methods=['POST'])
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

    # Вспомогательные методы остаются без изменений
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

    def _get_improved_template(self):
        """Улучшенный HTML шаблон с чанковой загрузкой"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Интерактивное обучение системы v2.0</title>
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

        .upload-area.uploading {
            border-color: #f39c12;
            background: #fef9e7;
        }

        .progress-container {
            margin: 20px 0;
            display: none;
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            width: 0%;
        }

        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 12px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
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

        .btn:hover:not(:disabled) {
            background: #5a67d8;
            transform: translateY(-2px);
        }

        .btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
            transform: none;
        }

        .btn.secondary {
            background: #e2e8f0;
            color: #4a5568;
        }

        .btn.secondary:hover:not(:disabled) {
            background: #cbd5e0;
        }

        .btn.success {
            background: #48bb78;
        }

        .btn.success:hover:not(:disabled) {
            background: #38a169;
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

        .bread-type-btn.defective {
            border-color: #e53e3e;
            color: #e53e3e;
        }

        .bread-type-btn.defective:hover {
            border-color: #c53030;
            background: #fed7d7;
            color: #c53030;
        }

        .bread-type-btn.defective.active {
            background: #e53e3e;
            color: white;
            border-color: #c53030;
        }

        .progress-section {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
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
            max-height: 300px;
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

        .file-list {
            margin: 15px 0;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }

        .file-item {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            cursor: pointer;
            transition: background 0.2s;
        }

        .file-item:hover {
            background: #f7fafc;
        }

        .file-item:last-child {
            border-bottom: none;
        }

        .file-name {
            font-weight: bold;
            color: #2d3748;
        }

        .file-details {
            font-size: 12px;
            color: #718096;
            margin-top: 2px;
        }

        .chunk-size-mb {
            font-size: 11px;
            color: #667eea;
            font-style: italic;
        }

        /* Стили для кастомных типов хлеба */
        #customBreadForm input {
            font-family: inherit;
            font-size: 14px;
        }

        #customBreadForm input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .bread-type-btn {
            position: relative;
        }

        .bread-type-btn .remove-btn {
            position: absolute;
            right: 5px;
            top: 5px;
            background: #e53e3e;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 12px;
            line-height: 18px;
            text-align: center;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .bread-type-btn:hover .remove-btn {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Интерактивное обучение системы v2.0</h1>
            <p>Загрузите MP4 видео большого размера и разметьте типы хлеба для обучения ИИ</p>
            <div class="chunk-size-mb">Поддержка файлов до 5GB с чанковой загрузкой</div>
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
                        <small style="color: #666;">Поддерживаются: MP4, AVI, MOV, MKV (до 5GB)</small>
                        <br><small style="color: #999;">Большие файлы загружаются по частям</small>
                    </div>

                    <div style="margin-top: 20px; position: relative; z-index: 3; pointer-events: auto;">
                        <button class="btn success" id="selectExistingVideo" style="width: 100%;">
                            📂 Выбрать уже загруженный файл
                        </button>
                    </div>
                </div>

                <div class="progress-container" id="progressContainer">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                        <div class="progress-text" id="progressText">0%</div>
                    </div>
                    <div style="margin-top: 10px; text-align: center;" id="progressDetails">
                        Подготовка к загрузке...
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
                        <button class="btn secondary" id="markAllObjects" style="margin-top: 10px;">🏷️ Пометить все как выбранный тип</button>
                    </div>
                </div>
            </div>

            <div class="control-panel">
                <h3>🎯 Управление обучением</h3>

                <div class="bread-types">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4>Тип хлеба:</h4>
                        <button class="btn secondary" id="addCustomBread" style="padding: 5px 10px; font-size: 12px;">
                            ➕ Добавить
                        </button>
                    </div>

                    <div id="breadTypesList">
                        <button class="bread-type-btn active" data-type="white_bread">🍞 Белый хлеб</button>
                        <button class="bread-type-btn" data-type="dark_bread">🍞 Черный хлеб</button>
                        <button class="bread-type-btn" data-type="baton">🥖 Батон</button>
                        <button class="bread-type-btn" data-type="molded_bread">📦 Хлеб в формах</button>
                        <button class="bread-type-btn defective" data-type="defective_bread">❌ Брак</button>
                    </div>

                    <div id="customBreadForm" class="hidden" style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <h5>Добавить новый тип хлеба:</h5>
                        <input type="text" id="customBreadName" placeholder="Например: Олександрівський, 0.7кг" 
                               style="width: 100%; padding: 8px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px;">
                        <div style="display: flex; gap: 10px;">
                            <button class="btn" id="saveCustomBread" style="flex: 1;">✅ Сохранить</button>
                            <button class="btn secondary" id="cancelCustomBread" style="flex: 1;">❌ Отмена</button>
                        </div>
                    </div>
                </div>

                <div class="detection-list" id="detectionList">
                    <p>Детекции появятся здесь</p>
                </div>

                <div class="progress-section">
                    <h4>📊 Прогресс обучения</h4>
                    <div class="progress-bar">
                        <div class="progress-fill" id="trainingProgressFill" style="width: 0%"></div>
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
                    <button class="btn success" id="loadExisting" style="margin-top: 10px;">📁 Загрузить файл</button>
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
        let uploadSessionId = null;
        let isUploading = false;
        let customBreadTypes = []; // Кастомные типы хлеба

        // Настройки чанковой загрузки
        const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB чанки

        // DOM элементы
        const uploadArea = document.getElementById('uploadArea');
        const videoInput = document.getElementById('videoInput');
        const videoSection = document.getElementById('videoSection');
        const videoFrame = document.getElementById('videoFrame');
        const frameSlider = document.getElementById('frameSlider');
        const frameInfo = document.getElementById('frameInfo');
        const detectionList = document.getElementById('detectionList');
        const statusMessages = document.getElementById('statusMessages');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressDetails = document.getElementById('progressDetails');

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            updateTrainingProgress();
            checkServerConnection();
        });

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

            // Предотвращаем открытие файла в браузере
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
            document.getElementById('markAllObjects').addEventListener('click', markAllObjects);

            // Новые обработчики
            document.getElementById('selectExistingVideo').addEventListener('click', selectExistingVideo);
            document.getElementById('addCustomBread').addEventListener('click', showCustomBreadForm);
            document.getElementById('saveCustomBread').addEventListener('click', saveCustomBread);
            document.getElementById('cancelCustomBread').addEventListener('click', hideCustomBreadForm);

            // Загрузка сохраненных кастомных типов хлеба
            loadCustomBreadTypes();
        }

        function handleDragEnter(e) {
            e.preventDefault();
            e.stopPropagation();
            if (!isUploading) uploadArea.classList.add('drag-over');
        }

        function handleDragOver(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        function handleDragLeave(e) {
            e.preventDefault();
            e.stopPropagation();
            if (!uploadArea.contains(e.relatedTarget)) {
                uploadArea.classList.remove('drag-over');
            }
        }

        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();

            uploadArea.classList.remove('drag-over');

            if (isUploading) {
                showStatus('Уже идет загрузка файла', 'warning');
                return;
            }

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('video/')) {
                    uploadVideoChunked(file);
                } else {
                    showStatus('Пожалуйста, выберите видео файл', 'error');
                }
            }
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file && !isUploading) {
                uploadVideoChunked(file);
            }
        }

        async function uploadVideoChunked(file) {
            if (isUploading) {
                showStatus('Уже идет загрузка', 'warning');
                return;
            }

            isUploading = true;
            uploadArea.classList.add('uploading');
            progressContainer.style.display = 'block';

            try {
                // 1. Инициализация загрузки
                showStatus('Инициализация загрузки...', 'info');
                progressDetails.textContent = `Подготовка файла ${file.name} (${(file.size / 1024 / 1024 / 1024).toFixed(2)} GB)`;

                const initResponse = await fetch('/start_chunked_upload', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        filename: file.name,
                        file_size: file.size
                    })
                });

                const initData = await initResponse.json();
                if (!initData.success) {
                    throw new Error(initData.error);
                }

                uploadSessionId = initData.session_id;
                showStatus('Начинается загрузка по частям...', 'info');

                // 2. Загрузка чанками
                const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

                for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                    const start = chunkIndex * CHUNK_SIZE;
                    const end = Math.min(start + CHUNK_SIZE, file.size);
                    const chunk = file.slice(start, end);

                    const formData = new FormData();
                    formData.append('session_id', uploadSessionId);
                    formData.append('chunk_index', chunkIndex);
                    formData.append('chunk', chunk);

                    const chunkResponse = await fetch('/upload_chunk', {
                        method: 'POST',
                        body: formData
                    });

                    const chunkData = await chunkResponse.json();
                    if (!chunkData.success) {
                        throw new Error('Ошибка загрузки чанка ' + chunkIndex);
                    }

                    // Обновляем прогресс
                    const progress = chunkData.progress;
                    progressFill.style.width = progress + '%';
                    progressText.textContent = progress.toFixed(1) + '%';
                    progressDetails.textContent = 
                        `Загружено ${(chunkData.uploaded_size / 1024 / 1024).toFixed(1)} MB из ${(file.size / 1024 / 1024).toFixed(1)} MB (часть ${chunkIndex + 1}/${totalChunks})`;

                    // Небольшая пауза между чанками
                    await new Promise(resolve => setTimeout(resolve, 50));
                }

                // 3. Завершение загрузки
                showStatus('Завершение загрузки...', 'info');
                progressDetails.textContent = 'Обработка видео...';

                const finishResponse = await fetch('/finish_upload', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: uploadSessionId})
                });

                const finishData = await finishResponse.json();
                if (finishData.success) {
                    totalFrames = finishData.total_frames;
                    frameSlider.max = totalFrames - 1;

                    uploadArea.classList.add('hidden');
                    videoSection.classList.remove('hidden');
                    progressContainer.style.display = 'none';

                    showStatus(finishData.message, 'success');
                    loadFrame(0);
                } else {
                    throw new Error(finishData.error);
                }

            } catch (error) {
                showStatus('Ошибка загрузки: ' + error.message, 'error');
                console.error('Upload error:', error);
            } finally {
                isUploading = false;
                uploadArea.classList.remove('uploading');
                uploadArea.classList.remove('drag-over');
                if (!totalFrames) {
                    progressContainer.style.display = 'none';
                }
                // Сбрасываем input
                videoInput.value = '';
            }
        }

        function loadExistingVideo() {
            // Используем новую функцию выбора видео
            selectExistingVideo();
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

                    uploadArea.classList.add('hidden');
                    videoSection.classList.remove('hidden');

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

        function testServerConnection() {
            showStatus('Тестирование подключения к серверу...', 'info');

            fetch('/get_training_progress')
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    } else {
                        throw new Error(`HTTP ${response.status}`);
                    }
                })
                .then(data => {
                    showStatus('✅ Сервер работает корректно!', 'success');
                })
                .catch(error => {
                    showStatus('❌ Ошибка подключения: ' + error.message, 'error');
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

            const defectiveCount = currentDetections.filter(d => d.is_defective).length;
            const totalCount = currentDetections.length;
            const defectRate = totalCount > 0 ? (defectiveCount / totalCount * 100).toFixed(1) : 0;

            let html = `
                <h5>Найденные объекты: ${totalCount}</h5>
                <div style="font-size: 12px; color: #666; margin-bottom: 10px;">
                    ✅ Норма: ${totalCount - defectiveCount} | ❌ Брак: ${defectiveCount} (${defectRate}%)
                </div>
            `;

            currentDetections.forEach((detection, index) => {
                const defectiveClass = detection.is_defective ? 'style="background: #fed7d7; border-left: 4px solid #e53e3e;"' : '';

                html += `
                    <div class="detection-item" ${defectiveClass} onclick="highlightDetection(${index})">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>ID ${detection.id}</strong><br>
                                Размер: ${detection.area}px²<br>
                                Уверенность: ${(detection.confidence * 100).toFixed(1)}%<br>
                                Тип: ${detection.type === 'unknown' ? 'Не определен' : detection.type}
                                ${detection.is_defective ? '<br><span style="color: #e53e3e;">❌ БРАК</span>' : ''}
                            </div>
                            <div style="display: flex; flex-direction: column; gap: 5px;">
                                <button class="btn" style="padding: 5px 10px; font-size: 12px;" 
                                        onclick="markAsDefective(${index}); event.stopPropagation();">
                                    ${detection.is_defective ? '✅ Норма' : '❌ Брак'}
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });
            detectionList.innerHTML = html;
        }

        function selectBreadType(type) {
            selectedBreadType = type;
            rebuildBreadTypesList(); // Обновляем активные кнопки
        }

        function saveCurrentAnnotation() {
            if (currentDetections.length === 0) {
                showStatus('Нет детекций для сохранения', 'error');
                return;
            }

            // Подсчитываем брак
            const defectiveCount = currentDetections.filter(d => d.is_defective).length;
            const totalCount = currentDetections.length;

            const annotationData = {
                frame_index: currentFrame,
                annotations: currentDetections,
                bread_type: selectedBreadType,
                defective_count: defectiveCount,
                total_count: totalCount,
                defect_rate: (defectiveCount / totalCount * 100).toFixed(1)
            };

            fetch('/annotate_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotationData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Аннотация сохранена! Брак: ${defectiveCount}/${totalCount} (${annotationData.defect_rate}%)`, 'success');
                    updateTrainingProgress();
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
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(extractData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(data.message, 'success');
                    updateTrainingProgress();
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

        function updateTrainingProgress() {
            fetch('/get_training_progress')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalAnnotations').textContent = data.total_annotations;

                    const progress = Math.min(data.total_annotations / 100 * 100, 100);
                    document.getElementById('trainingProgressFill').style.width = progress + '%';

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
            statusDiv.textContent = new Date().toLocaleTimeString() + ': ' + message;

            statusMessages.appendChild(statusDiv);

            // Скроллим к последнему сообщению
            statusMessages.scrollTop = statusMessages.scrollHeight;

            // Автоматическое удаление через 10 секунд
            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.remove();
                }
            }, 10000);
        }

        function highlightDetection(index) {
            console.log('Выбрана детекция:', currentDetections[index]);
        }

        function markAsDefective(index) {
            if (index >= 0 && index < currentDetections.length) {
                const detection = currentDetections[index];
                detection.is_defective = !detection.is_defective;

                // Обновляем отображение
                updateDetectionList();

                // Показываем статус
                const status = detection.is_defective ? 'помечен как БРАК' : 'помечен как НОРМА';
                showStatus(`Объект ID ${detection.id} ${status}`, 'info');

                console.log(`Detection ${detection.id} marked as ${detection.is_defective ? 'defective' : 'normal'}`);
            }
        }

        // Обновляем прогресс каждые 30 секунд
        setInterval(updateTrainingProgress, 30000);

        // ===== ФУНКЦИИ ДЛЯ РАБОТЫ С СУЩЕСТВУЮЩИМИ ВИДЕО =====

        function selectExistingVideo() {
            showStatus('Загрузка списка существующих файлов...', 'info');

            fetch('/list_uploaded_videos')
                .then(response => response.json())
                .then(data => {
                    if (data.videos && data.videos.length > 0) {
                        showVideoSelectionDialog(data.videos);
                    } else {
                        showStatus('В папке uploads нет загруженных видео файлов', 'warning');
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
                <h3 style="margin-bottom: 20px; color: #2d3748;">📂 Выберите видео для обучения</h3>
                <div style="margin-bottom: 20px; color: #718096; font-size: 14px;">
                    Найдено ${videos.length} загруженных видео файлов
                </div>
                <div class="file-list" style="max-height: 400px; overflow-y: auto;">
            `;

            videos.forEach((video, index) => {
                const sizeText = video.size_gb > 1 ? 
                    `${video.size_gb} GB` : 
                    `${video.size_mb} MB`;

                html += `
                    <div class="file-item" onclick="selectVideoFromDialog('${video.filename}')" 
                         style="padding: 15px; margin: 10px 0; border: 2px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: all 0.2s;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="file-name" style="font-weight: bold; color: #2d3748; margin-bottom: 5px;">
                                    🎬 ${video.filename}
                                </div>
                                <div class="file-details" style="font-size: 12px; color: #718096;">
                                    📦 ${sizeText} • 🗓️ ${video.modified}
                                </div>
                            </div>
                            <div style="color: #667eea; font-size: 24px;">▶️</div>
                        </div>
                    </div>
                `;
            });

            html += `
                </div>
                <div style="margin-top: 25px; text-align: right; border-top: 1px solid #e2e8f0; padding-top: 20px;">
                    <button class="btn secondary" onclick="closeVideoSelectionDialog()" style="margin-right: 10px;">❌ Отмена</button>
                    <button class="btn" onclick="refreshVideoList()">🔄 Обновить список</button>
                </div>
            `;

            dialog.innerHTML = html;
            modal.appendChild(dialog);
            document.body.appendChild(modal);

            // Добавляем стили для hover эффекта
            const fileItems = modal.querySelectorAll('.file-item');
            fileItems.forEach(item => {
                item.addEventListener('mouseenter', () => {
                    item.style.borderColor = '#667eea';
                    item.style.backgroundColor = '#f8f9ff';
                    item.style.transform = 'translateY(-2px)';
                });
                item.addEventListener('mouseleave', () => {
                    item.style.borderColor = '#e2e8f0';
                    item.style.backgroundColor = 'white';  
                    item.style.transform = 'translateY(0)';
                });
            });

            // Глобальные функции для модального окна
            window.selectVideoFromDialog = function(filename) {
                closeVideoSelectionDialog();
                loadSelectedVideo(filename);
            };

            window.closeVideoSelectionDialog = function() {
                document.body.removeChild(modal);
                delete window.selectVideoFromDialog;
                delete window.closeVideoSelectionDialog;
                delete window.refreshVideoList;
            };

            window.refreshVideoList = function() {
                closeVideoSelectionDialog();
                selectExistingVideo();
            };
        }

        // ===== ФУНКЦИИ ДЛЯ КАСТОМНЫХ ТИПОВ ХЛЕБА =====

        function loadCustomBreadTypes() {
            const savedTypes = localStorage.getItem('customBreadTypes');
            if (savedTypes) {
                try {
                    customBreadTypes = JSON.parse(savedTypes);
                    rebuildBreadTypesList();
                } catch (e) {
                    console.warn('Ошибка загрузки кастомных типов хлеба:', e);
                }
            }
        }

        function saveCustomBreadTypes() {
            localStorage.setItem('customBreadTypes', JSON.stringify(customBreadTypes));
        }

        function showCustomBreadForm() {
            document.getElementById('customBreadForm').classList.remove('hidden');
            document.getElementById('customBreadName').focus();
        }

        function hideCustomBreadForm() {
            document.getElementById('customBreadForm').classList.add('hidden');
            document.getElementById('customBreadName').value = '';
        }

        function saveCustomBread() {
            const name = document.getElementById('customBreadName').value.trim();

            if (!name) {
                showStatus('Введите название хлеба', 'warning');
                return;
            }

            // Проверяем на дубликаты
            const typeId = 'custom_' + name.toLowerCase().replace(/[^a-zA-Zа-яА-Я0-9]/g, '_');
            const existingType = customBreadTypes.find(t => t.id === typeId);

            if (existingType) {
                showStatus('Такой тип хлеба уже существует', 'warning');
                return;
            }

            // Добавляем новый тип
            const newType = {
                id: typeId,
                name: name,
                emoji: '🍞',
                created: new Date().toISOString()
            };

            customBreadTypes.push(newType);
            saveCustomBreadTypes();
            rebuildBreadTypesList();
            hideCustomBreadForm();

            showStatus(`Добавлен новый тип хлеба: "${name}"`, 'success');
        }

        function rebuildBreadTypesList() {
            const container = document.getElementById('breadTypesList');

            // Базовые типы
            let html = `
                <button class="bread-type-btn ${selectedBreadType === 'white_bread' ? 'active' : ''}" data-type="white_bread">🍞 Белый хлеб</button>
                <button class="bread-type-btn ${selectedBreadType === 'dark_bread' ? 'active' : ''}" data-type="dark_bread">🍞 Черный хлеб</button>
                <button class="bread-type-btn ${selectedBreadType === 'baton' ? 'active' : ''}" data-type="baton">🥖 Батон</button>
                <button class="bread-type-btn ${selectedBreadType === 'molded_bread' ? 'active' : ''}" data-type="molded_bread">📦 Хлеб в формах</button>
            `;

            // Кастомные типы
            customBreadTypes.forEach(type => {
                html += `
                    <button class="bread-type-btn ${selectedBreadType === type.id ? 'active' : ''}" 
                            data-type="${type.id}" 
                            style="position: relative;">
                        ${type.emoji} ${type.name}
                        <span onclick="removeCustomBread('${type.id}'); event.stopPropagation();" 
                              style="position: absolute; right: 5px; top: 5px; background: #e53e3e; color: white; border-radius: 50%; width: 18px; height: 18px; font-size: 12px; line-height: 18px; text-align: center; cursor: pointer;">
                            ×
                        </span>
                    </button>
                `;
            });

            // Брак в конце
            html += `<button class="bread-type-btn defective ${selectedBreadType === 'defective_bread' ? 'active' : ''}" data-type="defective_bread">❌ Брак</button>`;

            container.innerHTML = html;

            // Переназначаем обработчики
            container.querySelectorAll('.bread-type-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    if (!e.target.closest('span')) { // Если не кликнули на кнопку удаления
                        selectBreadType(e.target.dataset.type);
                    }
                });
            });
        }

        function removeCustomBread(typeId) {
            if (confirm('Удалить этот тип хлеба?')) {
                customBreadTypes = customBreadTypes.filter(t => t.id !== typeId);
                saveCustomBreadTypes();

                // Если удаляемый тип был выбран, переключаемся на белый хлеб
                if (selectedBreadType === typeId) {
                    selectedBreadType = 'white_bread';
                }

                rebuildBreadTypesList();
                showStatus('Тип хлеба удален', 'info');
            }
        }
    </script>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Запуск веб-приложения"""
        print(f"🚀 Запуск улучшенного интерфейса обучения на http://{host}:{port}")
        print("📁 Поддержка загрузки больших MP4 файлов до 5GB!")
        print("⚡ Чанковая загрузка для стабильности")

        # Увеличиваем таймауты для больших файлов
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        self.app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    app = ImprovedTrainingApp()
    app.run()