# web/training_module_advanced.py
"""Продвинутый обучающий модуль с отдельными HTML/CSS/JS файлами"""

from core.imports import *
from core.batch_training import BatchTrainingManager
import cv2
import os


class AdvancedTrainingModule:
    """Продвинутый модуль обучения с поддержкой больших файлов"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Настройка путей к шаблонам и статическим файлам
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
        print(*template_dir)
        print(*static_dir)

        self.app.template_folder = template_dir
        self.app.static_folder = static_dir
        self.app.static_url_path = '/static'

        # Конфигурация для больших файлов
        self.app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['TEMP_FOLDER'] = 'temp_uploads'
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        # Менеджер пакетного обучения
        self.batch_manager = BatchTrainingManager()

        # Состояние загрузки
        self.upload_sessions = {}

        # Текущее состояние обучения
        self.current_video = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.video_cap = None
        self.current_zones = {
            'entry_zone': None,
            'counting_zone': None,
            'exit_zone': None,
            'gray_zones': []
        }
        self.detected_objects = []
        self.training_data = []

        # Создание директорий
        for folder in ['uploads', 'temp_uploads', 'training_data', 'training_data/zones', 'training_data/batches']:
            os.makedirs(folder, exist_ok=True)

        # Проверяем наличие шаблонов
        self._check_templates()

        self._setup_routes()

    def _check_templates(self):
        """Проверка наличия шаблонов и статических файлов"""
        required_files = [
            ('templates/training.html', 'HTML шаблон'),
            ('static/css/training.css', 'CSS стили'),
            ('static/js/training.js', 'Основной JavaScript'),
            ('static/js/batch_training.js', 'JavaScript пакетного обучения')
        ]

        missing_files = []

        for file_path, description in required_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            if not os.path.exists(full_path):
                missing_files.append(f"{description}: {file_path}")

        if missing_files:
            print("⚠️ Отсутствуют файлы шаблонов:")
            for file in missing_files:
                print(f"   - {file}")
            print("📝 Создайте эти файлы или используйте встроенный HTML")
        else:
            print("✅ Все файлы шаблонов найдены")

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.route('/training')
        def training_interface():
            """Главный интерфейс обучения - теперь используем шаблон"""
            try:
                return render_template('training.html')
            except Exception as e:
                print(f"❌ Ошибка загрузки шаблона: {e}")
                # Fallback - возвращаем простой HTML
                return self._get_fallback_template()

        # === ЧАНКОВАЯ ЗАГРУЗКА ===

        @self.app.route('/api/training/start_upload', methods=['POST'])
        def start_chunked_upload():
            """Начало чанковой загрузки"""
            try:
                data = request.get_json()
                filename = secure_filename(data.get('filename', 'video.mp4'))
                file_size = data.get('file_size', 0)

                session_id = str(uuid.uuid4())
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

                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Загрузка инициализирована'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/upload_chunk', methods=['POST'])
        def upload_chunk():
            """Загрузка чанка файла"""
            try:
                session_id = request.form.get('session_id')
                chunk_index = int(request.form.get('chunk_index', 0))

                if session_id not in self.upload_sessions:
                    return jsonify({'success': False, 'error': 'Неверная сессия'}), 400

                session = self.upload_sessions[session_id]

                if 'chunk' not in request.files:
                    return jsonify({'success': False, 'error': 'Нет данных чанка'}), 400

                chunk_file = request.files['chunk']
                chunk_data = chunk_file.read()

                # Дописываем чанк к временному файлу
                with open(session['temp_path'], 'ab') as f:
                    f.write(chunk_data)

                session['uploaded_size'] += len(chunk_data)
                progress = (session['uploaded_size'] / session['file_size']) * 100 if session['file_size'] > 0 else 0

                return jsonify({
                    'success': True,
                    'progress': progress,
                    'uploaded_size': session['uploaded_size']
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/finish_upload', methods=['POST'])
        def finish_upload():
            """Завершение загрузки"""
            try:
                data = request.get_json()
                session_id = data.get('session_id')

                if session_id not in self.upload_sessions:
                    return jsonify({'success': False, 'error': 'Неверная сессия'}), 400

                session = self.upload_sessions[session_id]

                # Перемещаем файл в финальную папку
                final_path = os.path.join(self.app.config['UPLOAD_FOLDER'], session['filename'])
                os.rename(session['temp_path'], final_path)

                # Проверяем что файл видео
                if self._is_video_file(final_path):
                    success = self._load_video(final_path)

                    if success:
                        session['status'] = 'completed'

                        # Очищаем сессию через 5 минут
                        threading.Timer(300, lambda: self.upload_sessions.pop(session_id, None)).start()

                        return jsonify({
                            'success': True,
                            'message': f'Видео загружено: {self.total_frames} кадров',
                            'filename': session['filename'],
                            'total_frames': self.total_frames
                        })
                else:
                    return jsonify({'success': False, 'error': 'Неподдерживаемый формат файла'}), 400

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        # === УПРАВЛЕНИЕ ФАЙЛАМИ ===

        @self.app.route('/api/training/files')
        def list_files():
            """Список загруженных файлов"""
            try:
                files = []
                upload_dir = self.app.config['UPLOAD_FOLDER']

                if os.path.exists(upload_dir):
                    for filename in os.listdir(upload_dir):
                        filepath = os.path.join(upload_dir, filename)

                        if os.path.isfile(filepath):
                            ext = os.path.splitext(filename)[1].lower()
                            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']

                            if ext in video_extensions:
                                stat = os.stat(filepath)
                                duration = self._get_video_duration(filepath)

                                files.append({
                                    'name': filename,
                                    'size_bytes': stat.st_size,
                                    'size_gb': round(stat.st_size / (1024 ** 3), 2),
                                    'size_mb': round(stat.st_size / (1024 ** 2), 1),
                                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                    'duration': duration,
                                    'extension': ext
                                })

                files.sort(key=lambda x: x['modified'], reverse=True)
                return jsonify({'files': files})

            except Exception as e:
                print(f"❌ Ошибка сканирования файлов: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/files/<filename>/select', methods=['POST'])
        def select_file(filename):
            """Выбор файла для обучения"""
            try:
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_filename(filename))

                if os.path.exists(filepath):
                    success = self._load_video(filepath)
                    if success:
                        return jsonify({
                            'success': True,
                            'message': f'Видео загружено: {self.total_frames} кадров',
                            'total_frames': self.total_frames,
                            'filename': filename
                        })
                    else:
                        return jsonify({'success': False, 'error': 'Ошибка загрузки видео'}), 500
                else:
                    return jsonify({'success': False, 'error': 'Файл не найден'}), 404

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/files/<filename>', methods=['DELETE'])
        def delete_file(filename):
            """Удаление файла"""
            try:
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_filename(filename))
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'error': 'Файл не найден'}), 404
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/files/<filename>/rename', methods=['POST'])
        def rename_file(filename):
            """Переименование файла"""
            try:
                data = request.json
                new_name = secure_filename(data.get('new_name', ''))

                old_path = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_filename(filename))
                new_path = os.path.join(self.app.config['UPLOAD_FOLDER'], new_name)

                if os.path.exists(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    return jsonify({'success': True, 'new_name': new_name})
                else:
                    return jsonify({'success': False, 'error': 'Ошибка переименования'}), 400

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        # === РАБОТА С ВИДЕО И ЗОНАМИ ===

        @self.app.route('/api/training/zones/template', methods=['GET'])
        def get_zone_templates():
            """Получение шаблонов зон для разных типов печей"""
            templates = {
                'standard_oven': {
                    'name': 'Стандартная печь',
                    'zones': {
                        'entry_zone': {
                            'type': 'entry',
                            'points': [
                                {'x': 50, 'y': 50}, {'x': 200, 'y': 50},
                                {'x': 200, 'y': 150}, {'x': 50, 'y': 150}
                            ]
                        },
                        'counting_zone': {
                            'type': 'counting',
                            'points': [
                                {'x': 250, 'y': 200}, {'x': 900, 'y': 200},
                                {'x': 900, 'y': 600}, {'x': 250, 'y': 600}
                            ]
                        },
                        'exit_zone': {
                            'type': 'exit',
                            'points': [
                                {'x': 950, 'y': 400}, {'x': 1100, 'y': 400},
                                {'x': 1100, 'y': 500}, {'x': 950, 'y': 500}
                            ]
                        },
                        'gray_zones': []
                    }
                },
                'conveyor_oven': {
                    'name': 'Конвейерная печь',
                    'zones': {
                        'entry_zone': {
                            'type': 'entry',
                            'points': [
                                {'x': 30, 'y': 100}, {'x': 150, 'y': 100},
                                {'x': 150, 'y': 200}, {'x': 30, 'y': 200}
                            ]
                        },
                        'counting_zone': {
                            'type': 'counting',
                            'points': [
                                {'x': 200, 'y': 150}, {'x': 1000, 'y': 150},
                                {'x': 1000, 'y': 650}, {'x': 200, 'y': 650}
                            ]
                        },
                        'exit_zone': {
                            'type': 'exit',
                            'points': [
                                {'x': 1050, 'y': 350}, {'x': 1200, 'y': 350},
                                {'x': 1200, 'y': 450}, {'x': 1050, 'y': 450}
                            ]
                        },
                        'gray_zones': []
                    }
                }
            }
            return jsonify({'templates': templates})

        @self.app.route('/api/training/frame/<int:frame_index>')
        def get_frame(frame_index):
            """Получение кадра видео"""
            if self.video_cap is None:
                return jsonify({'success': False, 'error': 'Видео не загружено'}), 400

            try:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.video_cap.read()

                if ret:
                    self.current_frame_index = frame_index
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')

                    return jsonify({
                        'success': True,
                        'frame_data': f"data:image/jpeg;base64,{image_base64}",
                        'frame_index': frame_index,
                        'total_frames': self.total_frames
                    })
                else:
                    return jsonify({'success': False, 'error': 'Ошибка получения кадра'}), 500

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/zones/save', methods=['POST'])
        def save_zones_permanent():
            """Постоянное сохранение зон для камеры/печи"""
            try:
                data = request.json
                zones_data = data.get('zones', {})
                camera_id = data.get('camera_id', 'default')
                camera_name = data.get('camera_name', f'Камера {camera_id}')

                zones_dir = 'training_data/zones'
                os.makedirs(zones_dir, exist_ok=True)

                zones_file = os.path.join(zones_dir, f"camera_{camera_id}_zones.json")
                zones_config = {
                    'camera_id': camera_id,
                    'camera_name': camera_name,
                    'zones': zones_data,
                    'created': datetime.now().isoformat() if not os.path.exists(zones_file) else None,
                    'updated': datetime.now().isoformat()
                }

                if os.path.exists(zones_file):
                    try:
                        with open(zones_file, 'r', encoding='utf-8') as f:
                            existing_config = json.load(f)
                            zones_config['created'] = existing_config.get('created')
                    except:
                        pass

                with open(zones_file, 'w', encoding='utf-8') as f:
                    json.dump(zones_config, f, ensure_ascii=False, indent=2)

                self.current_zones = zones_data
                return jsonify({
                    'success': True,
                    'message': f'Зоны сохранены для {camera_name}',
                    'zones_file': zones_file,
                    'camera_id': camera_id
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/zones/load', methods=['POST'])
        def load_zones_permanent():
            """Загрузка сохраненных зон для камеры/печи"""
            try:
                data = request.json
                camera_id = data.get('camera_id', 'default')
                zones_file = f"training_data/zones/camera_{camera_id}_zones.json"

                if os.path.exists(zones_file):
                    with open(zones_file, 'r', encoding='utf-8') as f:
                        zones_config = json.load(f)

                    self.current_zones = zones_config.get('zones', {})
                    return jsonify({
                        'success': True,
                        'zones': self.current_zones,
                        'message': f'Зоны загружены для {zones_config.get("camera_name", f"камеры {camera_id}")}',
                        'camera_id': camera_id,
                        'camera_name': zones_config.get('camera_name')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'Зоны для камеры {camera_id} не найдены',
                        'zones': {},
                        'camera_id': camera_id
                    })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/detect_camera', methods=['POST'])
        def detect_camera_from_video():
            """Определение камеры по имени видео файла"""
            try:
                data = request.json
                video_name = data.get('video_name', '')

                camera_id = 'default'
                camera_name = 'Неизвестная камера'

                import re
                ch_match = re.search(r'CH(\d+)', video_name)
                if ch_match:
                    camera_id = f"ch{ch_match.group(1)}"
                    camera_name = f"Канал {ch_match.group(1)}"
                else:
                    ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', video_name)
                    if ip_match:
                        camera_id = ip_match.group(1).replace('.', '_')
                        camera_name = f"Камера {ip_match.group(1)}"
                    else:
                        base_name = video_name.split('_')[0] if '_' in video_name else video_name.split('.')[0]
                        camera_id = base_name.lower()
                        camera_name = f"Камера {base_name}"

                return jsonify({
                    'success': True,
                    'camera_id': camera_id,
                    'camera_name': camera_name,
                    'video_name': video_name
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'camera_id': 'default',
                    'camera_name': 'Камера по умолчанию',
                    'error': str(e)
                })

        # === ДЕТЕКЦИЯ И АННОТАЦИИ ===

        @self.app.route('/api/training/detect')
        def detect_objects():
            """Детекция объектов на текущем кадре"""
            if self.video_cap is None:
                return jsonify({'success': False, 'error': 'Видео не загружено'}), 400

            try:
                detected = self._detect_bread_objects()
                return jsonify({
                    'success': True,
                    'objects': detected
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/save_annotation', methods=['POST'])
        def save_annotation():
            """Сохранение аннотации объекта"""
            try:
                data = request.json
                annotation = {
                    'object_id': data.get('object_id'),
                    'bbox': data.get('bbox'),
                    'frame_index': self.current_frame_index,
                    'product_info': {
                        'guid': data.get('guid', ''),
                        'sku_code': data.get('sku_code', ''),
                        'product_name': data.get('product_name', ''),
                        'category': data.get('category', 'bread')
                    },
                    'zones': self.current_zones,
                    'timestamp': datetime.now().isoformat(),
                    'is_validated': data.get('is_validated', False)
                }

                self.training_data.append(annotation)
                success = self._save_training_data()

                if success:
                    return jsonify({
                        'success': True,
                        'annotation_id': len(self.training_data) - 1,
                        'total_annotations': len(self.training_data)
                    })
                else:
                    return jsonify({'success': False, 'error': 'Ошибка сохранения файла'}), 500

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/data')
        def get_training_data():
            """Получение данных обучения"""
            try:
                return jsonify({
                    'zones': self.current_zones,
                    'training_data': self.training_data,
                    'total_annotations': len(self.training_data),
                    'current_video': os.path.basename(self.current_video) if self.current_video else None
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        # === API ДЛЯ ПАКЕТНОГО ОБУЧЕНИЯ ===
        # (Все API endpoints из batch_training_api.py)

        @self.app.route('/api/training/batch/create', methods=['POST'])
        def create_batch():
            """Создание новой партии для обучения"""
            try:
                data = request.json
                batch_info = {
                    'product_name': data.get('product_name', ''),
                    'sku_code': data.get('sku_code', ''),
                    'category': data.get('category', 'bread'),
                    'batch_size_estimate': data.get('batch_size_estimate', 1500),
                    'production_date': data.get('production_date', datetime.now().isoformat()),
                    'operator': data.get('operator', 'system'),
                    'notes': data.get('notes', '')
                }

                batch_id = self.batch_manager.create_batch(batch_info)

                return jsonify({
                    'success': True,
                    'batch_id': batch_id,
                    'message': f'Партия {batch_id} создана успешно'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/set_template', methods=['POST'])
        def set_batch_template():
            """Установка эталонных объектов для партии"""
            try:
                data = request.json
                template_objects = data.get('template_objects', [])

                if not template_objects:
                    return jsonify({'success': False, 'error': 'Не выбраны эталонные объекты'}), 400

                success = self.batch_manager.set_batch_template(template_objects)

                if success:
                    return jsonify({
                        'success': True,
                        'message': f'Эталон установлен на основе {len(template_objects)} объектов'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Не удалось установить эталон'}), 500

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/start_auto', methods=['POST'])
        def start_auto_training():
            """Запуск автоматического обучения"""
            try:
                success = self.batch_manager.start_auto_training()

                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Автоматическое обучение запущено'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Эталон не установлен'}), 400

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/process_frame', methods=['POST'])
        def process_frame_auto():
            """Автоматическая обработка кадра с детекцией"""
            try:
                detected = self._detect_bread_objects()

                if not detected:
                    return jsonify({
                        'success': True,
                        'results': {'processed': 0, 'stop_required': False}
                    })

                frame_data = {
                    'frame_index': self.current_frame_index,
                    'video_file': self.current_video,
                    'timestamp': datetime.now().isoformat()
                }

                results = self.batch_manager.process_detected_objects(detected, frame_data)

                return jsonify({
                    'success': True,
                    'results': results,
                    'detected_objects': detected
                })

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/anomalies')
        def get_pending_anomalies():
            """Получение списка аномалий, ожидающих разрешения"""
            try:
                anomalies = self.batch_manager.get_pending_anomalies()
                return jsonify({
                    'success': True,
                    'anomalies': anomalies
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/resolve_anomaly', methods=['POST'])
        def resolve_anomaly():
            """Разрешение аномалии оператором"""
            try:
                data = request.json
                anomaly_id = data.get('anomaly_id')
                resolution = data.get('resolution', {})

                if anomaly_id is None:
                    return jsonify({'success': False, 'error': 'Не указан ID аномалии'}), 400

                valid_actions = ['add_to_good', 'mark_as_defect', 'ignore']
                if resolution.get('action') not in valid_actions:
                    return jsonify({'success': False, 'error': 'Неверное действие'}), 400

                success = self.batch_manager.resolve_anomaly(anomaly_id, resolution)

                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Аномалия разрешена'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Не удалось разрешить аномалию'}), 500

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/statistics')
        def get_batch_statistics():
            """Получение статистики текущей партии"""
            try:
                stats = self.batch_manager.get_batch_statistics()
                return jsonify({
                    'success': True,
                    'statistics': stats
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/batch/stop', methods=['POST'])
        def stop_auto_training():
            """Остановка автоматического обучения"""
            try:
                self.batch_manager.auto_training_active = False
                if self.batch_manager.current_batch:
                    self.batch_manager.current_batch['status'] = 'stopped_manual'
                    self.batch_manager._save_batch()

                return jsonify({
                    'success': True,
                    'message': 'Автоматическое обучение остановлено'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===

    def _is_video_file(self, filepath):
        """Проверка что файл является видео"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        return os.path.splitext(filepath)[1].lower() in video_extensions

    def _get_video_duration(self, filepath):
        """Получение длительности видео"""
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                return round(duration, 1)
        except:
            pass
        return 0

    def _load_video(self, filepath):
        """Загрузка видео файла"""
        try:
            if self.video_cap:
                self.video_cap.release()

            self.video_cap = cv2.VideoCapture(filepath)
            if not self.video_cap.isOpened():
                return False

            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_index = 0
            self.current_video = filepath

            return True

        except Exception as e:
            print(f"Ошибка загрузки видео: {e}")
            return False

    def _detect_bread_objects(self):
        """Детекция объектов хлеба на текущем кадре"""
        if self.video_cap is None:
            return []

        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            ret, frame = self.video_cap.read()

            if not ret:
                return []

            # Простая детекция по цвету (временная эмуляция)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_bread = np.array([8, 50, 50])
            upper_bread = np.array([25, 255, 255])

            mask = cv2.inRange(hsv, lower_bread, upper_bread)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_objects = []

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if 2000 < area < 25000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    if 0.5 < aspect_ratio < 3.0:
                        if self._is_in_counting_zone(x + w // 2, y + h // 2):
                            detected_objects.append({
                                'id': f"bread_{i}",
                                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                'area': int(area),
                                'confidence': 0.85,
                                'center': {'x': int(x + w // 2), 'y': int(y + h // 2)}
                            })

            self.detected_objects = detected_objects
            return detected_objects

        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return []

    def _is_in_counting_zone(self, x, y):
        """Проверка нахождения точки в зоне подсчета"""
        counting_zone = self.current_zones.get('counting_zone')
        if not counting_zone:
            return True

        if 'points' in counting_zone and len(counting_zone['points']) >= 4:
            xs = [p['x'] for p in counting_zone['points']]
            ys = [p['y'] for p in counting_zone['points']]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            return min_x <= x <= max_x and min_y <= y <= max_y

        return True

    def _save_training_data(self):
        """Сохранение данных обучения"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_data/annotations_{timestamp}.json"

            data_to_save = {
                'video_file': os.path.basename(self.current_video) if self.current_video else None,
                'total_frames': self.total_frames,
                'current_frame_index': self.current_frame_index,
                'zones': self.current_zones,
                'annotations': self.training_data,
                'created': datetime.now().isoformat(),
                'total_objects': len(self.training_data)
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения данных обучения: {e}")
            return False

    def _get_fallback_template(self):
        """Простой fallback HTML если шаблоны не найдены"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Обучение - Fallback</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 2rem; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; }
        .error { background: #fed7d7; color: #742a2a; padding: 1rem; border-radius: 4px; margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="error">
            ⚠️ <strong>Файлы шаблонов не найдены!</strong><br>
            Создайте файлы HTML/CSS/JS в папках web/templates/ и web/static/
        </div>
        <h1>🧠 Обучение системы распознавания</h1>
        <p>Этот простой интерфейс работает, но для полной функциональности создайте файлы шаблонов.</p>
        <a href="/">← Вернуться на главную</a>
    </div>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Запуск модуля"""
        self.app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    module = AdvancedTrainingModule()
    module.run(debug=True)