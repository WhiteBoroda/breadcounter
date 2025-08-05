# web/training_module_advanced.py
"""Продвинутый обучающий модуль с чанковой загрузкой и управлением файлами"""

from core.imports import *
import cv2


class AdvancedTrainingModule:
    """Продвинутый модуль обучения с поддержкой больших файлов"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Конфигурация для больших файлов
        self.app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['TEMP_FOLDER'] = 'temp_uploads'
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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
        for folder in ['uploads', 'temp_uploads', 'training_data']:
            os.makedirs(folder, exist_ok=True)

        self._setup_routes()

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.route('/training')
        def training_interface():
            return render_template_string(self._get_training_template())

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
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/upload_chunk', methods=['POST'])
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

                return jsonify({
                    'success': True,
                    'progress': progress,
                    'uploaded_size': session['uploaded_size']
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/finish_upload', methods=['POST'])
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
                    return jsonify({'error': 'Неподдерживаемый формат файла'}), 400

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # === УПРАВЛЕНИЕ ФАЙЛАМИ ===

        @self.app.route('/api/training/files')
        def list_files():
            """Список загруженных файлов"""
            try:
                files = []
                upload_dir = self.app.config['UPLOAD_FOLDER']

                print(f"🔍 Сканирование папки: {upload_dir}")
                print(f"   Папка существует: {os.path.exists(upload_dir)}")

                if os.path.exists(upload_dir):
                    all_files = os.listdir(upload_dir)
                    print(f"   Всего файлов в папке: {len(all_files)}")

                    for filename in all_files:
                        filepath = os.path.join(upload_dir, filename)
                        print(f"   Проверяем файл: {filename}")

                        if os.path.isfile(filepath):
                            # Проверяем расширение файла
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
                                print(f"     ✅ Добавлен видео файл: {filename}")
                            else:
                                print(f"     ⏭️  Пропускаем (не видео): {filename}")
                        else:
                            print(f"     ⏭️  Пропускаем (не файл): {filename}")
                else:
                    print(f"❌ Папка {upload_dir} не существует")
                    os.makedirs(upload_dir, exist_ok=True)
                    print(f"✅ Создана папка {upload_dir}")

                files.sort(key=lambda x: x['modified'], reverse=True)
                print(f"📊 Итого найдено видео файлов: {len(files)}")

                return jsonify({'files': files})

            except Exception as e:
                print(f"❌ Ошибка сканирования файлов: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

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
                        return jsonify({'error': 'Ошибка загрузки видео'}), 500
                else:
                    return jsonify({'error': 'Файл не найден'}), 404

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/files/<filename>', methods=['DELETE'])
        def delete_file(filename):
            """Удаление файла"""
            try:
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], secure_filename(filename))
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return jsonify({'success': True})
                else:
                    return jsonify({'error': 'Файл не найден'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

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
                    return jsonify({'error': 'Ошибка переименования'}), 400

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # === РАБОТА С ВИДЕО И ОБУЧЕНИЕМ ===

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
                                {'x': 50, 'y': 50},
                                {'x': 200, 'y': 50},
                                {'x': 200, 'y': 150},
                                {'x': 50, 'y': 150}
                            ]
                        },
                        'counting_zone': {
                            'type': 'counting',
                            'points': [
                                {'x': 250, 'y': 200},
                                {'x': 900, 'y': 200},
                                {'x': 900, 'y': 600},
                                {'x': 250, 'y': 600}
                            ]
                        },
                        'exit_zone': {
                            'type': 'exit',
                            'points': [
                                {'x': 950, 'y': 400},
                                {'x': 1100, 'y': 400},
                                {'x': 1100, 'y': 500},
                                {'x': 950, 'y': 500}
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
                                {'x': 30, 'y': 100},
                                {'x': 150, 'y': 100},
                                {'x': 150, 'y': 200},
                                {'x': 30, 'y': 200}
                            ]
                        },
                        'counting_zone': {
                            'type': 'counting',
                            'points': [
                                {'x': 200, 'y': 150},
                                {'x': 1000, 'y': 150},
                                {'x': 1000, 'y': 650},
                                {'x': 200, 'y': 650}
                            ]
                        },
                        'exit_zone': {
                            'type': 'exit',
                            'points': [
                                {'x': 1050, 'y': 350},
                                {'x': 1200, 'y': 350},
                                {'x': 1200, 'y': 450},
                                {'x': 1050, 'y': 450}
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
                print(f"❌ Ошибка получения кадра {frame_index}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            """Получение кадра видео"""
            if self.video_cap is None:
                return jsonify({'error': 'Видео не загружено'}), 400

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
                    return jsonify({'error': 'Ошибка получения кадра'}), 500

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/zones/save', methods=['POST'])
        def save_zones_permanent():
            """Постоянное сохранение зон для камеры/печи"""
            try:
                data = request.json
                zones_data = data.get('zones', {})
                camera_id = data.get('camera_id', 'default')
                camera_name = data.get('camera_name', f'Камера {camera_id}')

                # Создаем папку для зон если не существует
                zones_dir = 'training_data/zones'
                os.makedirs(zones_dir, exist_ok=True)

                # Сохраняем зоны для конкретной камеры/печи
                zones_file = os.path.join(zones_dir, f"camera_{camera_id}_zones.json")

                zones_config = {
                    'camera_id': camera_id,
                    'camera_name': camera_name,
                    'zones': zones_data,
                    'created': datetime.now().isoformat() if not os.path.exists(zones_file) else None,
                    'updated': datetime.now().isoformat()
                }

                # Если файл существует, сохраняем дату создания
                if os.path.exists(zones_file):
                    with open(zones_file, 'r', encoding='utf-8') as f:
                        existing_config = json.load(f)
                        zones_config['created'] = existing_config.get('created')

                with open(zones_file, 'w', encoding='utf-8') as f:
                    json.dump(zones_config, f, ensure_ascii=False, indent=2)

                # Обновляем текущие зоны
                self.current_zones = zones_data

                print(f"💾 Зоны сохранены для камеры {camera_id}: {zones_file}")

                return jsonify({
                    'success': True,
                    'message': f'Зоны сохранены для {camera_name}',
                    'zones_file': zones_file,
                    'camera_id': camera_id
                })

            except Exception as e:
                print(f"❌ Ошибка сохранения зон: {e}")
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

                    print(f"📂 Зоны загружены для камеры {camera_id}: {zones_file}")

                    return jsonify({
                        'success': True,
                        'zones': self.current_zones,
                        'message': f'Зоны загружены для {zones_config.get("camera_name", f"камеры {camera_id}")}',
                        'camera_id': camera_id,
                        'camera_name': zones_config.get('camera_name')
                    })
                else:
                    print(f"⚠️ Файл зон не найден для камеры {camera_id}: {zones_file}")
                    return jsonify({
                        'success': False,
                        'message': f'Зоны для камеры {camera_id} не найдены',
                        'zones': {},
                        'camera_id': camera_id
                    })

            except Exception as e:
                print(f"❌ Ошибка загрузки зон: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/training/detect_camera', methods=['POST'])
        def detect_camera_from_video():
            """Определение камеры по имени видео файла"""
            try:
                data = request.json
                video_name = data.get('video_name', '')

                # Пытаемся извлечь IP камеры из имени файла
                # Примеры: "20250802_144412_CH32_20250720015826-20250720030517.mp4"
                # или "camera_192.168.1.100_video.mp4"

                camera_id = 'default'
                camera_name = 'Неизвестная камера'

                # Попытка 1: извлечь из имени файла шаблон CH{number}
                import re
                ch_match = re.search(r'CH(\d+)', video_name)
                if ch_match:
                    camera_id = f"ch{ch_match.group(1)}"
                    camera_name = f"Канал {ch_match.group(1)}"
                else:
                    # Попытка 2: извлечь IP адрес
                    ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', video_name)
                    if ip_match:
                        camera_id = ip_match.group(1).replace('.', '_')
                        camera_name = f"Камера {ip_match.group(1)}"
                    else:
                        # Попытка 3: использовать первую часть имени файла
                        base_name = video_name.split('_')[0] if '_' in video_name else video_name.split('.')[0]
                        camera_id = base_name.lower()
                        camera_name = f"Камера {base_name}"

                print(f"🔍 Определена камера для видео '{video_name}': {camera_id} ({camera_name})")

                return jsonify({
                    'success': True,
                    'camera_id': camera_id,
                    'camera_name': camera_name,
                    'video_name': video_name
                })

            except Exception as e:
                print(f"❌ Ошибка определения камеры: {e}")
                return jsonify({
                    'success': False,
                    'camera_id': 'default',
                    'camera_name': 'Камера по умолчанию',
                    'error': str(e)
                })

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
                print(f"❌ Ошибка получения кадра {frame_index}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            """Получение шаблонов зон для разных типов печей"""
            templates = {
                'standard_oven': {
                    'name': 'Стандартная печь',
                    'zones': {
                        'entry_zone': {
                            'type': 'entry',
                            'points': [
                                {'x': 50, 'y': 50},
                                {'x': 200, 'y': 50},
                                {'x': 200, 'y': 150},
                                {'x': 50, 'y': 150}
                            ]
                        },
                        'counting_zone': {
                            'type': 'counting',
                            'points': [
                                {'x': 250, 'y': 200},
                                {'x': 900, 'y': 200},
                                {'x': 900, 'y': 600},
                                {'x': 250, 'y': 600}
                            ]
                        },
                        'exit_zone': {
                            'type': 'exit',
                            'points': [
                                {'x': 950, 'y': 400},
                                {'x': 1100, 'y': 400},
                                {'x': 1100, 'y': 500},
                                {'x': 950, 'y': 500}
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
                                {'x': 30, 'y': 100},
                                {'x': 150, 'y': 100},
                                {'x': 150, 'y': 200},
                                {'x': 30, 'y': 200}
                            ]
                        },
                        'counting_zone': {
                            'type': 'counting',
                            'points': [
                                {'x': 200, 'y': 150},
                                {'x': 1000, 'y': 150},
                                {'x': 1000, 'y': 650},
                                {'x': 200, 'y': 650}
                            ]
                        },
                        'exit_zone': {
                            'type': 'exit',
                            'points': [
                                {'x': 1050, 'y': 350},
                                {'x': 1200, 'y': 350},
                                {'x': 1200, 'y': 450},
                                {'x': 1050, 'y': 450}
                            ]
                        },
                        'gray_zones': []
                    }
                }
            }

            return jsonify({'templates': templates})

        @self.app.route('/api/training/detect')
        def detect_objects():
            """Детекция объектов на текущем кадре"""
            if self.video_cap is None:
                return jsonify({'error': 'Видео не загружено'}), 400

            try:
                detected = self._detect_bread_objects()
                return jsonify({
                    'success': True,
                    'objects': detected
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/save_annotation', methods=['POST'])
        def save_annotation():
            """Сохранение аннотации объекта"""
            try:
                data = request.json
                print(f"🔄 Получен запрос на сохранение аннотации: {data.get('object_id')}")

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

                # Сохраняем в файл
                success = self._save_training_data()

                if success:
                    print(f"✅ Аннотация сохранена: {len(self.training_data)} всего")
                    return jsonify({
                        'success': True,
                        'annotation_id': len(self.training_data) - 1,
                        'total_annotations': len(self.training_data)
                    })
                else:
                    return jsonify({'success': False, 'error': 'Ошибка сохранения файла'}), 500

            except Exception as e:
                print(f"❌ Ошибка сохранения аннотации: {e}")
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
                print(f"❌ Ошибка получения данных обучения: {e}")
                return jsonify({'error': str(e)}), 500

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
            # Получаем текущий кадр
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            ret, frame = self.video_cap.read()

            if not ret:
                return []

            # Простая детекция по цвету (как в предыдущем коде)
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

            print(f"💾 Данные обучения сохранены в {filename}")
            print(f"   Аннотаций: {len(self.training_data)}")
            print(
                f"   Зон: {sum(1 for zone in self.current_zones.values() if zone and (not isinstance(zone, list) or len(zone) > 0))}")

            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения данных обучения: {e}")
            return False

    def _get_training_template(self):
        """HTML шаблон продвинутого обучающего интерфейса"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Обучение системы распознавания</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; }
        .header { background: #2d3748; color: white; padding: 1rem; }
        .container { max-width: 1600px; margin: 1rem auto; padding: 0 1rem; }
        .main-grid { display: grid; grid-template-columns: 280px 1fr 320px; gap: 1rem; }

        /* Адаптивность */
        @media (max-width: 1400px) {
            .main-grid { grid-template-columns: 240px 1fr 280px; gap: 0.8rem; }
            .container { padding: 0 0.5rem; }
        }

        @media (max-width: 1200px) {
            .main-grid { 
                grid-template-columns: 1fr; 
                max-width: 100%;
            }
            .panel { margin-bottom: 1rem; }
            .video-controls { flex-wrap: wrap; gap: 0.5rem; }
        }

        @media (max-width: 800px) {
            .zone-controls { 
                display: flex; 
                flex-wrap: wrap; 
                gap: 0.3rem; 
            }
            .zone-btn { 
                padding: 0.5rem 0.8rem; 
                font-size: 12px; 
                margin: 0.1rem;
            }
        }
        .panel { background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }

        /* Загрузка файлов */
        .upload-area { 
            border: 2px dashed #cbd5e0; border-radius: 8px; padding: 1.5rem; text-align: center; 
            margin-bottom: 1rem; cursor: pointer; transition: all 0.3s; min-height: 100px;
            display: flex; flex-direction: column; justify-content: center;
        }
        .upload-area:hover { border-color: #3182ce; background: #f7fafc; }
        .upload-area.uploading { border-color: #d69e2e; background: #fffaf0; }

        /* Прогресс загрузки */
        .progress-container { display: none; margin: 1rem 0; }
        .progress { width: 100%; height: 16px; background: #e2e8f0; border-radius: 8px; overflow: hidden; }
        .progress-bar { height: 100%; background: #3182ce; transition: width 0.3s; }
        .progress-text { text-align: center; margin-top: 0.5rem; font-size: 12px; }

        /* Список файлов */
        .file-list { max-height: 350px; overflow-y: auto; }
        .file-item { 
            background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; 
            padding: 0.6rem; margin-bottom: 0.4rem; cursor: pointer; transition: all 0.2s;
        }
        .file-item:hover { background: #edf2f7; border-color: #cbd5e0; }
        .file-item.selected { background: #ebf8ff; border-color: #3182ce; }
        .file-info { font-size: 11px; color: #718096; margin-top: 0.25rem; }
        .file-controls { margin-top: 0.4rem; display: flex; gap: 0.2rem; }

        /* Видео панель */
        .video-panel { display: none; }
        .video-panel.active { display: block; }
        .canvas-container { 
            position: relative; max-width: 100%; overflow: auto; margin-bottom: 1rem; 
            border: 1px solid #e2e8f0; border-radius: 6px;
        }
        .canvas-overlay { position: absolute; top: 0; left: 0; pointer-events: auto; }

        /* Контролы видео */
        .video-controls { 
            display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem; 
            padding: 0.75rem; background: #f7fafc; border-radius: 6px;
        }
        .frame-slider { flex: 1; }

        /* Зоны */
        .zone-controls { margin-bottom: 1rem; }
        .zone-btn { 
            padding: 0.5rem 1rem; margin: 0.25rem; border: none; border-radius: 4px; 
            cursor: pointer; font-size: 14px; transition: all 0.2s;
        }
        .zone-btn.active { transform: scale(1.05); box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        .zone-btn.entry { background: #38a169; color: white; }
        .zone-btn.counting { background: #d69e2e; color: white; }
        .zone-btn.exit { background: #e53e3e; color: white; }
        .zone-btn.gray { background: #718096; color: white; }

        /* Объекты */
        .object-list { max-height: 300px; overflow-y: auto; }
        .object-item { 
            background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; 
            padding: 1rem; margin-bottom: 0.5rem; cursor: pointer; transition: all 0.2s;
        }
        .object-item:hover { background: #edf2f7; }
        .object-item.selected { border-color: #3182ce; background: #ebf8ff; }
        .object-item.annotated { 
            background: #f0fff4; 
            border-color: #68d391; 
        }
        .object-item.annotated:hover { 
            background: #e6fffa; 
        }
        .object-item.pending { 
            background: #fffaf0; 
            border-color: #fbd38d; 
        }

        /* Формы */
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.25rem; font-weight: 500; }
        .form-group input, .form-group select { 
            width: 100%; padding: 0.5rem; border: 1px solid #e2e8f0; border-radius: 4px; 
        }

        /* Кнопки */
        .btn { 
            padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; 
            font-size: 14px; margin: 0.25rem; transition: all 0.2s;
        }
        .btn:hover { transform: translateY(-1px); }
        .btn-sm { padding: 0.25rem 0.5rem; font-size: 12px; }
        .btn-primary { background: #3182ce; color: white; }
        .btn-success { background: #38a169; color: white; }
        .btn-danger { background: #e53e3e; color: white; }
        .btn-secondary { background: #718096; color: white; }
        .btn-warning { background: #d69e2e; color: white; }

        /* Статус */
        .status { padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0; }
        .status.success { background: #c6f6d5; color: #22543d; border-left: 4px solid #38a169; }
        .status.error { background: #fed7d7; color: #742a2a; border-left: 4px solid #e53e3e; }
        .status.info { background: #bee3f8; color: #2a4365; border-left: 4px solid #3182ce; }
        .status.warning { background: #fefcbf; color: #744210; border-left: 4px solid #d69e2e; }

        /* Статистика */
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem; }
        .stat-item { background: #f7fafc; padding: 0.75rem; border-radius: 6px; text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #2b6cb0; }
        .stat-label { font-size: 12px; color: #718096; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Обучение системы распознавания хлеба</h1>
        <a href="/" style="color: #cbd5e0; text-decoration: none;">← Вернуться на главную</a>
    </div>

    <div class="container">
        <div class="main-grid">
            <!-- Левая панель - управление файлами -->
            <div class="panel">
                <h3>📁 Управление файлами</h3>

                <!-- Загрузка -->
                <div class="upload-area" id="uploadArea">
                    <p>📤 Перетащите видео сюда или кликните для выбора</p>
                    <input type="file" id="fileInput" accept="video/*" style="display: none;">
                </div>

                <!-- Прогресс загрузки -->
                <div class="progress-container" id="progressContainer">
                    <div class="progress">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                    <div class="progress-text" id="progressText">0%</div>
                    <div class="progress-text" id="progressDetails"></div>
                </div>

                <!-- Список файлов -->
                <div class="file-list" id="fileList"></div>

                <!-- Статистика -->
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="totalFiles">0</div>
                        <div class="stat-label">Файлов</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="totalSize">0</div>
                        <div class="stat-label">GB</div>
                    </div>
                </div>
            </div>

            <!-- Центральная панель - видео -->
            <div class="panel">
                <div id="videoPanel" class="video-panel">
                    <h3>🎬 Работа с видео</h3>

                    <!-- Контролы видео -->
                    <div class="video-controls">
                        <button class="btn btn-secondary" onclick="previousFrame()">⏮️</button>
                        <input type="range" id="frameSlider" class="frame-slider" min="0" max="100" value="0">
                        <button class="btn btn-secondary" onclick="nextFrame()">⏭️</button>
                        <span id="frameInfo">0 / 0</span>
                    </div>

                    <!-- Информация о камере -->
                    <div id="cameraInfo" style="background: #f0f4f8; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; font-size: 12px; color: #2d3748; display: none;">
                        <strong>📹 Камера:</strong> <span id="cameraName">Не определена</span> 
                        <span style="color: #718096;">(<span id="cameraId">unknown</span>)</span>
                    </div>

                    <!-- Зоны -->
                    <div class="zone-controls">
                        <div style="margin-bottom: 0.5rem; font-weight: bold; color: #2d3748;">🎯 Разметка зон детекции</div>

                        <!-- Шаблоны зон -->
                        <div style="margin-bottom: 0.75rem; padding: 0.5rem; background: #edf2f7; border-radius: 4px;">
                            <div style="font-size: 12px; color: #4a5568; margin-bottom: 0.25rem;">📋 Шаблоны зон:</div>
                            <button class="btn btn-sm btn-secondary" onclick="loadZoneTemplate('standard_oven')">🏭 Стандартная печь</button>
                            <button class="btn btn-sm btn-secondary" onclick="loadZoneTemplate('conveyor_oven')">🏗️ Конвейерная</button>
                        </div>

                        <!-- Кнопки создания зон -->
                        <button class="zone-btn entry" onclick="setZoneMode('entry')">🟢 Вход</button>
                        <button class="zone-btn counting" onclick="setZoneMode('counting')">🟡 Подсчет</button>
                        <button class="zone-btn exit" onclick="setZoneMode('exit')">🔴 Выход</button>
                        <button class="zone-btn gray" onclick="setZoneMode('gray')">⚫ Серая</button>

                        <!-- Управление зонами -->
                        <div style="margin-top: 0.75rem;">
                            <button class="btn btn-primary" onclick="saveZones()">💾 Сохранить зоны</button>
                            <button class="btn btn-primary" onclick="detectObjects()">🔍 Найти объекты</button>
                            <button class="btn btn-danger" onclick="clearZones()">🗑️ Очистить</button>
                        </div>
                    </div>

                    <!-- Canvas для видео -->
                    <div class="canvas-container">
                        <canvas id="videoCanvas"></canvas>
                        <canvas id="overlayCanvas" class="canvas-overlay"></canvas>
                    </div>
                </div>

                <div id="status"></div>
            </div>

            <!-- Правая панель - объекты и аннотации -->
            <div class="panel">
                <h3>📋 Обнаруженные объекты</h3>
                <div id="objectsList" class="object-list"></div>

                <!-- Форма аннотации -->
                <div id="annotationForm" style="display: none;">
                    <h3>📝 Описание продукта</h3>
                    <div class="form-group">
                        <label>GUID:</label>
                        <input type="text" id="productGuid" readonly>
                    </div>
                    <div class="form-group">
                        <label>Код SKU:</label>
                        <input type="text" id="productSku" placeholder="Введите SKU">
                    </div>
                    <div class="form-group">
                        <label>Наименование:</label>
                        <input type="text" id="productName" placeholder="Название продукта">
                    </div>
                    <div class="form-group">
                        <label>Категория:</label>
                        <select id="productCategory">
                            <option value="bread">Хлеб</option>
                            <option value="bun">Булочки</option>
                            <option value="loaf">Батон</option>
                            <option value="pastry">Выпечка</option>
                        </select>
                    </div>
                    <button class="btn btn-success" onclick="saveAnnotation()">💾 Сохранить</button>
                    <button class="btn btn-secondary" onclick="cancelAnnotation()">❌ Отмена</button>
                </div>

                <!-- Статистика обучения -->
                <div style="margin-top: 2rem;">
                    <h3>📊 Статистика</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="annotatedCount">0</div>
                            <div class="stat-label">Аннотаций</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="zonesCount">0</div>
                            <div class="stat-label">Зон</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Константы
        const CHUNK_SIZE = 1024 * 1024; // 1MB чанки

        // Глобальные переменные
        let isUploading = false;
        let uploadSessionId = null;
        let currentVideo = null;
        let currentCamera = {id: 'default', name: 'Неизвестная камера'};
        let totalFrames = 0;
        let currentFrame = 0;
        let currentZoneMode = null;
        let zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
        let detectedObjects = [];
        let selectedObject = null;
        let isDrawing = false;
        let startPoint = null;

        // DOM элементы (будут инициализированы после загрузки DOM)
        let uploadArea, fileInput, progressContainer, progressBar, progressText, progressDetails;
        let fileList, videoPanel, frameSlider, frameInfo, videoCanvas, overlayCanvas;
        let ctx, overlayCtx;

        console.log('📝 Глобальные переменные инициализированы');

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            loadFileList();
        });

        function setupEventListeners() {
            console.log('🎧 Настройка обработчиков событий...');

            // Элементы для drag & drop
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const frameSlider = document.getElementById('frameSlider');

            if (uploadArea && fileInput) {
                // Drag & Drop
                uploadArea.addEventListener('dragenter', handleDragEnter);
                uploadArea.addEventListener('dragover', handleDragOver);
                uploadArea.addEventListener('dragleave', handleDragLeave);
                uploadArea.addEventListener('drop', handleDrop);
                uploadArea.addEventListener('click', () => {
                    console.log('📁 Клик по области загрузки');
                    fileInput.click();
                });

                fileInput.addEventListener('change', handleFileSelect);
                console.log('✅ Upload обработчики настроены');
            } else {
                console.error('❌ Элементы upload не найдены');
            }

            // Слайдер кадров
            if (frameSlider) {
                frameSlider.addEventListener('input', function() {
                    loadFrame(parseInt(this.value));
                });
                console.log('✅ Frame slider настроен');
            } else {
                console.error('❌ Frame slider не найден');
            }

            console.log('🎧 Обработчики событий настроены');
        }

        function handleDragEnter(e) {
            e.preventDefault();
            if (!isUploading) uploadArea.style.borderColor = '#3182ce';
        }

        function handleDragOver(e) {
            e.preventDefault();
        }

        function handleDragLeave(e) {
            e.preventDefault();
            if (!uploadArea.contains(e.relatedTarget)) {
                uploadArea.style.borderColor = '#cbd5e0';
            }
        }

        function handleDrop(e) {
            e.preventDefault();
            uploadArea.style.borderColor = '#cbd5e0';

            if (isUploading) return;

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                uploadVideoChunked(files[0]);
            } else {
                showStatus('Выберите видео файл', 'error');
            }
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file && !isUploading) {
                uploadVideoChunked(file);
            }
        }

        async function uploadVideoChunked(file) {
            if (isUploading) return;

            isUploading = true;
            uploadArea.classList.add('uploading');
            progressContainer.style.display = 'block';

            try {
                showStatus('Начинается загрузка...', 'info');

                // 1. Инициализация
                const initResponse = await fetch('/api/training/start_upload', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        filename: file.name,
                        file_size: file.size
                    })
                });

                const initData = await initResponse.json();
                if (!initData.success) throw new Error(initData.error);

                uploadSessionId = initData.session_id;

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

                    const chunkResponse = await fetch('/api/training/upload_chunk', {
                        method: 'POST',
                        body: formData
                    });

                    const chunkData = await chunkResponse.json();
                    if (!chunkData.success) throw new Error('Ошибка загрузки чанка');

                    // Обновляем прогресс
                    const progress = chunkData.progress;
                    progressBar.style.width = progress + '%';
                    progressText.textContent = progress.toFixed(1) + '%';
                    progressDetails.textContent = 
                        `Загружено ${(chunkData.uploaded_size / 1024 / 1024).toFixed(1)} MB из ${(file.size / 1024 / 1024).toFixed(1)} MB`;

                    await new Promise(resolve => setTimeout(resolve, 10));
                }

                // 3. Завершение
                const finishResponse = await fetch('/api/training/finish_upload', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: uploadSessionId})
                });

                const finishData = await finishResponse.json();
                if (finishData.success) {
                    showStatus(finishData.message, 'success');
                    totalFrames = finishData.total_frames;
                    frameSlider.max = totalFrames - 1;
                    frameInfo.textContent = `0 / ${totalFrames}`;

                    videoPanel.classList.add('active');
                    loadFrame(0);
                    loadFileList();
                } else {
                    throw new Error(finishData.error);
                }

            } catch (error) {
                showStatus('Ошибка загрузки: ' + error.message, 'error');
            } finally {
                isUploading = false;
                uploadArea.classList.remove('uploading');
                progressContainer.style.display = 'none';
                uploadSessionId = null;
            }
        }

        function loadFileList() {
            console.log('🔄 Загрузка списка файлов...');

            fetch('/api/training/files')
                .then(response => {
                    console.log('📡 Ответ сервера:', response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('📋 Данные с сервера:', data);

                    if (data.files) {
                        console.log(`📊 Найдено файлов: ${data.files.length}`);
                        renderFileList(data.files);
                        updateFileStats(data.files);
                    } else {
                        console.warn('⚠️ Нет поля files в ответе');
                        renderFileList([]);
                        updateFileStats([]);
                    }
                })
                .catch(error => {
                    console.error('❌ Ошибка загрузки списка файлов:', error);
                    showStatus('Ошибка загрузки файлов: ' + error.message, 'error');
                    renderFileList([]);
                    updateFileStats([]);
                });
        }

        function renderFileList(files) {
            const fileList = document.getElementById('fileList');
            console.log(`🎨 Отрисовка ${files.length} файлов`);

            if (!fileList) {
                console.error('❌ Элемент fileList не найден');
                return;
            }

            if (files.length === 0) {
                fileList.innerHTML = '<p style="color: #718096; text-align: center; padding: 1rem;">Видео файлы не найдены.<br>Загрузите видео для обучения</p>';
                return;
            }

            fileList.innerHTML = files.map((file, index) => {
                const sizeText = file.size_gb > 1 ? 
                    `${file.size_gb} GB` : 
                    `${file.size_mb} MB`;

                const dateText = new Date(file.modified).toLocaleDateString('ru-RU');

                console.log(`📄 Рендерим файл: ${file.name} (${sizeText})`);

                return `
                    <div class="file-item" onclick="selectFile('${file.name}')">
                        <div><strong>${file.name}</strong></div>
                        <div class="file-info">
                            ${sizeText} • ${file.duration}s • ${dateText}
                        </div>
                        <div class="file-controls">
                            <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); selectFile('${file.name}')">
                                📂 Открыть
                            </button>
                            <button class="btn btn-sm btn-warning" onclick="event.stopPropagation(); renameFile('${file.name}')">
                                ✏️ 
                            </button>
                            <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteFile('${file.name}')">
                                🗑️
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            console.log('✅ Файлы отрисованы в DOM');
        }

        function updateFileStats(files) {
            console.log(`📊 Обновление статистики для ${files.length} файлов`);

            const totalFilesElement = document.getElementById('totalFiles');
            const totalSizeElement = document.getElementById('totalSize');

            if (totalFilesElement) {
                totalFilesElement.textContent = files.length;
                console.log(`📁 Обновлено количество файлов: ${files.length}`);
            } else {
                console.error('❌ Элемент totalFiles не найден');
            }

            if (totalSizeElement) {
                const totalGB = files.reduce((sum, file) => sum + (file.size_gb || 0), 0);
                totalSizeElement.textContent = totalGB.toFixed(1);
                console.log(`💾 Обновлен размер: ${totalGB.toFixed(1)} GB`);
            } else {
                console.error('❌ Элемент totalSize не найден');
            }
        }

        function selectFile(filename) {
            console.log(`📂 Выбор файла: ${filename}`);

            // Сначала определяем камеру по имени файла
            fetch('/api/training/detect_camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video_name: filename})
            })
            .then(response => response.json())
            .then(cameraData => {
                if (cameraData.success) {
                    currentCamera = {
                        id: cameraData.camera_id,
                        name: cameraData.camera_name
                    };
                    console.log(`🎥 Определена камера: ${currentCamera.name} (${currentCamera.id})`);

                    // Теперь загружаем видео
                    return fetch(`/api/training/files/${filename}/select`, {method: 'POST'});
                } else {
                    console.warn('⚠️ Не удалось определить камеру, используем по умолчанию');
                    currentCamera = {id: 'default', name: 'Камера по умолчанию'};
                    return fetch(`/api/training/files/${filename}/select`, {method: 'POST'});
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`${data.message} (${currentCamera.name})`, 'success');
                    totalFrames = data.total_frames;
                    frameSlider.max = totalFrames - 1;
                    frameInfo.textContent = `0 / ${totalFrames}`;

                    videoPanel.classList.add('active');
                    loadFrame(0);

                    // Показываем информацию о камере
                    const cameraInfo = document.getElementById('cameraInfo');
                    const cameraNameSpan = document.getElementById('cameraName');
                    const cameraIdSpan = document.getElementById('cameraId');

                    if (cameraInfo && cameraNameSpan && cameraIdSpan) {
                        cameraInfo.style.display = 'block';
                        cameraNameSpan.textContent = currentCamera.name;
                        cameraIdSpan.textContent = currentCamera.id;
                    }

                    // Выделяем выбранный файл
                    document.querySelectorAll('.file-item').forEach(item => {
                        item.classList.toggle('selected', item.textContent.includes(filename));
                    });

                    // Загружаем сохраненные зоны для этой камеры
                    loadZonesForCamera(currentCamera.id, currentCamera.name);

                } else {
                    showStatus('Ошибка: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка выбора файла: ' + error.message, 'error');
            });
        }

        function loadZonesForCamera(cameraId, cameraName) {
            console.log(`🔄 Загрузка зон для камеры: ${cameraName} (${cameraId})`);

            fetch('/api/training/zones/load', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera_id: cameraId})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success && data.zones) {
                    zones = data.zones;
                    console.log('✅ Зоны загружены для камеры:', zones);
                    redrawOverlay();
                    updateZonesCount();
                    showStatus(`Зоны загружены для ${cameraName}`, 'success');
                } else {
                    console.log(`ℹ️ Сохраненные зоны для камеры ${cameraName} не найдены`);
                    zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
                    showStatus(`Зоны для ${cameraName} не найдены - создайте новые`, 'info');
                }
            })
            .catch(error => {
                console.error('❌ Ошибка загрузки зон:', error);
                zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
                showStatus('Ошибка загрузки зон - используем пустые', 'warning');
            });
        }

        function saveZones() {
            if (Object.values(zones).every(zone => !zone || (Array.isArray(zone) && zone.length === 0))) {
                showStatus('Сначала нарисуйте хотя бы одну зону', 'warning');
                return;
            }

            showStatus('Сохранение зон...', 'info');

            // Сохраняем зоны для текущей камеры
            fetch('/api/training/zones/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    zones: zones,
                    camera_id: currentCamera.id,
                    camera_name: currentCamera.name
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Зоны сохранены для ${currentCamera.name}`, 'success');
                    updateZonesCount();

                    // Автоматическая детекция после сохранения зон
                    if (zones.counting_zone) {
                        setTimeout(() => {
                            detectObjects();
                        }, 500);
                    }
                } else {
                    showStatus('Ошибка сохранения зон: ' + (data.error || 'неизвестная ошибка'), 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка сети при сохранении зон: ' + error.message, 'error');
            });
        }

        function autoSaveZones() {
            // Автосохранение зон для текущей камеры
            if (!currentCamera.id) return;

            fetch('/api/training/zones/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    zones: zones,
                    camera_id: currentCamera.id,
                    camera_name: currentCamera.name
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`✅ Зоны автосохранены для ${currentCamera.name}`);
                } else {
                    console.warn('⚠️ Ошибка автосохранения зон:', data.error);
                }
            })
            .catch(error => {
                console.warn('⚠️ Ошибка автосохранения зон:', error);
            });
        }

        function loadZoneTemplate(templateName) {
            showStatus('Загрузка шаблона зон...', 'info');

            fetch('/api/training/zones/template')
                .then(response => response.json())
                .then(data => {
                    if (data.templates && data.templates[templateName]) {
                        zones = data.templates[templateName].zones;
                        redrawOverlay();
                        updateZonesCount();
                        showStatus(`Шаблон "${data.templates[templateName].name}" загружен`, 'success');
                    } else {
                        showStatus('Шаблон не найден', 'error');
                    }
                })
                .catch(error => {
                    showStatus('Ошибка загрузки шаблона: ' + error.message, 'error');
                });
        }

        function deleteFile(filename) {
            if (confirm(`Удалить файл ${filename}?`)) {
                fetch(`/api/training/files/${filename}`, {method: 'DELETE'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showStatus('Файл удален', 'success');
                            loadFileList();
                        } else {
                            showStatus('Ошибка удаления', 'error');
                        }
                    });
            }
        }

        function renameFile(filename) {
            const newName = prompt('Новое имя файла:', filename);
            if (newName && newName !== filename) {
                fetch(`/api/training/files/${filename}/rename`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({new_name: newName})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showStatus('Файл переименован', 'success');
                        loadFileList();
                    } else {
                        showStatus('Ошибка переименования', 'error');
                    }
                });
            }
        }

        function loadFrame(frameIndex) {
            if (frameIndex < 0 || frameIndex >= totalFrames) return;

            showStatus(`Загрузка кадра ${frameIndex}...`, 'info');

            fetch(`/api/training/frame/${frameIndex}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentFrame = frameIndex;
                        frameSlider.value = frameIndex;
                        frameInfo.textContent = `${frameIndex} / ${totalFrames}`;

                        const img = new Image();
                        img.onload = function() {
                            // Используем оригинальные размеры изображения
                            videoCanvas.width = img.width;
                            videoCanvas.height = img.height;
                            overlayCanvas.width = img.width;
                            overlayCanvas.height = img.height;

                            // Рисуем изображение без масштабирования
                            ctx.drawImage(img, 0, 0);
                            redrawOverlay();

                            setupCanvasEvents();
                        };
                        img.src = data.frame_data;

                        showStatus(`Кадр ${frameIndex} загружен`, 'success');
                    } else {
                        showStatus('Ошибка загрузки кадра: ' + (data.error || 'неизвестная ошибка'), 'error');
                    }
                })
                .catch(error => {
                    showStatus('Ошибка сети при загрузке кадра: ' + error.message, 'error');
                });
        }

        function autoSaveZones() {
            // Получаем имя текущего видео файла
            const selectedFile = document.querySelector('.file-item.selected');
            if (!selectedFile) return;

            const videoName = selectedFile.querySelector('strong').textContent.replace('.mp4', '');

            fetch('/api/training/zones/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    zones: zones,
                    video_name: videoName
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('✅ Зоны автосохранены');
                } else {
                    console.warn('⚠️ Ошибка автосохранения зон:', data.error);
                }
            })
            .catch(error => {
                console.warn('⚠️ Ошибка автосохранения зон:', error);
            });
        }

        function previousFrame() {
            if (currentFrame > 0) loadFrame(currentFrame - 1);
        }

        function nextFrame() {
            if (currentFrame < totalFrames - 1) loadFrame(currentFrame + 1);
        }

        function setupCanvasEvents() {
            // Удаляем предыдущие обработчики, если они есть
            overlayCanvas.removeEventListener('mousedown', startDrawing);
            overlayCanvas.removeEventListener('mousemove', draw);
            overlayCanvas.removeEventListener('mouseup', stopDrawing);

            // Добавляем новые обработчики
            overlayCanvas.addEventListener('mousedown', startDrawing);
            overlayCanvas.addEventListener('mousemove', draw);
            overlayCanvas.addEventListener('mouseup', stopDrawing);

            console.log('✅ События canvas настроены');
        }

        function startDrawing(e) {
            if (!currentZoneMode) return;

            isDrawing = true;
            const rect = overlayCanvas.getBoundingClientRect();
            startPoint = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        }

        function draw(e) {
            if (!isDrawing || !currentZoneMode) return;

            const rect = overlayCanvas.getBoundingClientRect();
            const currentPoint = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };

            redrawOverlay();

            overlayCtx.strokeStyle = getZoneColor(currentZoneMode);
            overlayCtx.lineWidth = 2;
            overlayCtx.strokeRect(
                startPoint.x,
                startPoint.y,
                currentPoint.x - startPoint.x,
                currentPoint.y - startPoint.y
            );
        }

        function stopDrawing(e) {
            if (!isDrawing || !currentZoneMode) return;

            isDrawing = false;
            const rect = overlayCanvas.getBoundingClientRect();
            const endPoint = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };

            const zone = {
                type: currentZoneMode,
                points: [
                    startPoint,
                    {x: endPoint.x, y: startPoint.y},
                    endPoint,
                    {x: startPoint.x, y: endPoint.y}
                ]
            };

            if (currentZoneMode === 'gray') {
                zones.gray_zones.push(zone);
            } else {
                zones[currentZoneMode + '_zone'] = zone;
            }

            saveZones();
            redrawOverlay();
            updateZonesCount();
        }

        function setZoneMode(mode) {
            // Сбрасываем предыдущий режим если тот же
            if (currentZoneMode === mode) {
                currentZoneMode = null;
                showStatus('Режим разметки выключен', 'info');
            } else {
                currentZoneMode = mode;
                showStatus(`Режим разметки: ${mode} - нарисуйте прямоугольник мышкой`, 'info');
            }

            // Обновляем стили кнопок
            document.querySelectorAll('.zone-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            if (currentZoneMode) {
                const activeBtn = event.target;
                activeBtn.classList.add('active');
            }
        }

        function saveZones() {
            if (Object.values(zones).every(zone => !zone || (Array.isArray(zone) && zone.length === 0))) {
                showStatus('Сначала нарисуйте хотя бы одну зону', 'warning');
                return;
            }

            showStatus('Сохранение зон...', 'info');

            fetch('/api/training/zones', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(zones)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Зоны сохранены успешно', 'success');
                    updateZonesCount();

                    if (data.detected_objects) {
                        detectedObjects = data.detected_objects;
                        renderObjectsList();
                        redrawOverlay();
                        showStatus(`Зоны сохранены + найдено объектов: ${detectedObjects.length}`, 'success');
                    }
                } else {
                    showStatus('Ошибка сохранения зон: ' + (data.error || 'неизвестная ошибка'), 'error');
                }
            })
            .catch(error => {
                showStatus('Ошибка сети при сохранении зон: ' + error.message, 'error');
            });
        }

                    console.log('Ошибка получения данных обучения:', error);
                });
        }

        function updateAnnotatedCount() {
            // Подсчитываем аннотированные объекты в текущей сессии
            const currentSessionCount = detectedObjects.filter(obj => obj.annotated).length;

            // Получаем общее количество из API
            fetch('/api/training/data')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    const serverCount = data.total_annotations || 0;
                    console.log(`Аннотации: сервер=${serverCount}, сессия=${currentSessionCount}`);
                    document.getElementById('annotatedCount').textContent = serverCount;
                })
                .catch(error => {
                    // Если API недоступно, показываем локальный счетчик
                    console.warn('Не удалось получить счетчик аннотаций:', error);
                    document.getElementById('annotatedCount').textContent = currentSessionCount;
                });
        }

        function saveAnnotation() {
            if (selectedObject === null) {
                showStatus('Выберите объект для аннотации', 'error');
                return;
            }

            const obj = detectedObjects[selectedObject];
            const guid = document.getElementById('productGuid').value.trim();
            const sku = document.getElementById('productSku').value.trim();
            const name = document.getElementById('productName').value.trim();
            const category = document.getElementById('productCategory').value;

            if (!sku || !name) {
                showStatus('Заполните SKU и наименование продукта', 'error');
                return;
            }

            const annotation = {
                object_id: obj.id,
                bbox: obj.bbox,
                guid: guid,
                sku_code: sku,
                product_name: name,
                category: category,
                is_validated: true
            };

            showStatus('Сохранение аннотации...', 'info');
            console.log('Отправляем аннотацию:', annotation);

            fetch('/api/training/save_annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotation)
            })
            .then(response => {
                console.log('Ответ сервера:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Данные ответа:', data);
                if (data.success) {
                    showStatus(`Аннотация для "${name}" сохранена успешно`, 'success');

                    // Помечаем объект как аннотированный
                    obj.annotated = true;
                    obj.annotation_data = annotation;

                    // Обновляем счетчик сразу
                    if (data.total_annotations) {
                        document.getElementById('annotatedCount').textContent = data.total_annotations;
                    } else {
                        updateAnnotatedCount();
                    }

                    // Очищаем форму и снимаем выделение
                    cancelAnnotation();

                    // Перерисовываем с учетом аннотированного объекта
                    renderObjectsList();
                    redrawOverlay();
                } else {
                    showStatus('Ошибка сохранения: ' + (data.error || 'неизвестная ошибка'), 'error');
                }
            })
            .catch(error => {
                console.error('Ошибка сохранения аннотации:', error);
                showStatus('Ошибка сети при сохранении: ' + error.message, 'error');
            });
        }

        function getZoneColor(zoneType) {
            const colors = {
                'entry': '#38a169',
                'counting': '#d69e2e', 
                'exit': '#e53e3e',
                'gray': '#718096'
            };
            return colors[zoneType] || '#3182ce';
        }

        function redrawOverlay() {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            // Рисуем зоны
            Object.entries(zones).forEach(([key, zone]) => {
                if (key === 'gray_zones') {
                    zone.forEach(grayZone => drawZone(grayZone, 'gray'));
                } else if (zone) {
                    const zoneType = key.replace('_zone', '');
                    drawZone(zone, zoneType);
                }
            });

            // Рисуем объекты
            detectedObjects.forEach((obj, index) => {
                drawObject(obj, index === selectedObject);
            });
        }

        function drawZone(zone, type) {
            if (!zone.points || zone.points.length < 4) return;

            overlayCtx.strokeStyle = getZoneColor(type);
            overlayCtx.lineWidth = 2;
            overlayCtx.setLineDash([5, 5]);

            overlayCtx.beginPath();
            overlayCtx.moveTo(zone.points[0].x, zone.points[0].y);
            zone.points.forEach(point => {
                overlayCtx.lineTo(point.x, point.y);
            });
            overlayCtx.closePath();
            overlayCtx.stroke();

            overlayCtx.setLineDash([]);
        }

        function drawObject(obj, isSelected) {
            const bbox = obj.bbox;

            // Цвета для разных состояний объекта
            let strokeColor, fillColor;
            if (obj.annotated) {
                strokeColor = '#22543d';  // Зеленый для аннотированных
                fillColor = 'rgba(34, 84, 61, 0.1)';
            } else if (isSelected) {
                strokeColor = '#3182ce';  // Синий для выбранного
                fillColor = 'rgba(49, 130, 206, 0.1)';
            } else {
                strokeColor = '#e53e3e';  // Красный для обычных
                fillColor = 'rgba(229, 62, 62, 0.05)';
            }

            // Рисуем прямоугольник
            overlayCtx.strokeStyle = strokeColor;
            overlayCtx.fillStyle = fillColor;
            overlayCtx.lineWidth = isSelected ? 3 : 2;

            overlayCtx.fillRect(bbox.x, bbox.y, bbox.width, bbox.height);
            overlayCtx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

            // Подпись объекта
            overlayCtx.fillStyle = strokeColor;
            overlayCtx.font = 'bold 12px Arial';

            let label = obj.id;
            if (obj.annotated && obj.annotation_data) {
                label += ` (${obj.annotation_data.sku_code})`;
            }

            // Фон для текста
            const textMetrics = overlayCtx.measureText(label);
            const textWidth = textMetrics.width + 8;
            const textHeight = 16;

            overlayCtx.fillStyle = strokeColor;
            overlayCtx.fillRect(bbox.x, bbox.y - textHeight - 2, textWidth, textHeight);

            overlayCtx.fillStyle = 'white';
            overlayCtx.fillText(label, bbox.x + 4, bbox.y - 6);

            // Иконка для аннотированных объектов
            if (obj.annotated) {
                overlayCtx.fillStyle = '#22543d';
                overlayCtx.font = 'bold 14px Arial';
                overlayCtx.fillText('✓', bbox.x + bbox.width - 20, bbox.y + 16);
            }
        }

        function stopDrawing(e) {
            if (!isDrawing || !currentZoneMode) return;

            isDrawing = false;
            const rect = overlayCanvas.getBoundingClientRect();
            const endPoint = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };

            // Проверяем что зона достаточно большая
            const minSize = 20;
            if (Math.abs(endPoint.x - startPoint.x) < minSize || Math.abs(endPoint.y - startPoint.y) < minSize) {
                showStatus('Зона слишком маленькая - нарисуйте больший прямоугольник', 'warning');
                redrawOverlay();
                return;
            }

            const zone = {
                type: currentZoneMode,
                points: [
                    startPoint,
                    {x: endPoint.x, y: startPoint.y},
                    endPoint,
                    {x: startPoint.x, y: endPoint.y}
                ]
            };

            if (currentZoneMode === 'gray') {
                zones.gray_zones.push(zone);
            } else {
                zones[currentZoneMode + '_zone'] = zone;
            }

            redrawOverlay();
            updateZonesCount();

            showStatus(`Зона "${currentZoneMode}" создана`, 'success');

            // Автосохранение зон
            setTimeout(() => {
                autoSaveZones();
            }, 1000);
        }

        function loadFrame(frameIndex) {
            if (frameIndex < 0 || frameIndex >= totalFrames) return;

            showStatus(`Загрузка кадра ${frameIndex}...`, 'info');

            fetch(`/api/training/frame/${frameIndex}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentFrame = frameIndex;
                        frameSlider.value = frameIndex;
                        frameInfo.textContent = `${frameIndex} / ${totalFrames}`;

                        const img = new Image();
                        img.onload = function() {
                            // Используем оригинальные размеры изображения
                            videoCanvas.width = img.width;
                            videoCanvas.height = img.height;
                            overlayCanvas.width = img.width;
                            overlayCanvas.height = img.height;

                            // Рисуем изображение без масштабирования
                            ctx.drawImage(img, 0, 0);
                            redrawOverlay();

                            setupCanvasEvents();
                        };
                        img.src = data.frame_data;

                        showStatus(`Кадр ${frameIndex} загружен`, 'success');
                    } else {
                        showStatus('Ошибка загрузки кадра: ' + (data.error || 'неизвестная ошибка'), 'error');
                    }
                })
                .catch(error => {
                    showStatus('Ошибка сети при загрузке кадра: ' + error.message, 'error');
                });
        }

        function clearZones() {
            if (Object.values(zones).every(zone => !zone || (Array.isArray(zone) && zone.length === 0))) {
                showStatus('Зоны уже очищены', 'info');
                return;
            }

            if (confirm('Удалить все размеченные зоны?')) {
                zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
                redrawOverlay();
                updateZonesCount();
                showStatus('Все зоны очищены', 'success');
            }
        }

        function detectObjects() {
            showStatus('Поиск объектов...', 'info');

            fetch('/api/training/detect')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        detectedObjects = data.objects || [];
                        renderObjectsList();
                        redrawOverlay();
                        showStatus(`Найдено объектов: ${detectedObjects.length}`, 'success');
                    } else {
                        showStatus('Ошибка детекции: ' + (data.error || 'неизвестная ошибка'), 'error');
                    }
                })
                .catch(error => {
                    showStatus('Ошибка сети при детекции: ' + error.message, 'error');
                });
        }

        function renderObjectsList() {
            const list = document.getElementById('objectsList');

            if (detectedObjects.length === 0) {
                list.innerHTML = '<p style="color: #718096; text-align: center; padding: 1rem;">Объекты не найдены.<br>Нажмите "Найти объекты"</p>';
                return;
            }

            list.innerHTML = detectedObjects.map((obj, index) => {
                const statusIcon = obj.annotated ? '✅' : '📝';
                const statusText = obj.annotated ? 'Аннотирован' : 'Требует аннотации';
                const statusClass = obj.annotated ? 'annotated' : 'pending';

                return `<div class="object-item ${statusClass}" onclick="selectObject(${index})">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>${obj.id}</strong>
                        <span style="font-size: 12px;">${statusIcon} ${statusText}</span>
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">
                        Позиция: ${obj.bbox.x}, ${obj.bbox.y}<br>
                        Размер: ${obj.bbox.width}×${obj.bbox.height}<br>
                        Уверенность: ${(obj.confidence * 100).toFixed(1)}%
                        ${obj.annotated && obj.annotation_data ? 
                            `<br><strong>SKU:</strong> ${obj.annotation_data.sku_code}<br><strong>Продукт:</strong> ${obj.annotation_data.product_name}` : ''
                        }
                    </div>
                </div>`;
            }).join('');
        }

        function selectObject(index) {
            selectedObject = index;

            document.querySelectorAll('.object-item').forEach((item, i) => {
                item.classList.toggle('selected', i === index);
            });

            redrawOverlay();

            // Показываем форму только для неаннотированных объектов
            const obj = detectedObjects[index];
            if (!obj.annotated) {
                document.getElementById('annotationForm').style.display = 'block';
                document.getElementById('productGuid').value = generateGUID();

                // Очищаем предыдущие данные
                document.getElementById('productSku').value = '';
                document.getElementById('productName').value = '';
                document.getElementById('productCategory').value = 'bread';
            } else {
                // Для аннотированных объектов показываем данные
                showStatus(`Объект уже аннотирован: ${obj.annotation_data.product_name}`, 'info');
            }
        }

        function cancelAnnotation() {
            document.getElementById('annotationForm').style.display = 'none';
            selectedObject = null;
            redrawOverlay();
        }

        function saveAnnotation() {
            if (selectedObject === null) {
                showStatus('Выберите объект', 'error');
                return;
            }

            const obj = detectedObjects[selectedObject];
            const annotation = {
                object_id: obj.id,
                bbox: obj.bbox,
                guid: document.getElementById('productGuid').value,
                sku_code: document.getElementById('productSku').value,
                product_name: document.getElementById('productName').value,
                category: document.getElementById('productCategory').value,
                is_validated: true
            };

            fetch('/api/training/save_annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotation)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Аннотация сохранена', 'success');
                    updateAnnotatedCount();
                    cancelAnnotation();
                } else {
                    showStatus('Ошибка сохранения: ' + data.error, 'error');
                }
            });
        }

        function updateZonesCount() {
            let count = 0;
            if (zones.entry_zone) count++;
            if (zones.counting_zone) count++;
            if (zones.exit_zone) count++;
            count += zones.gray_zones.length;

            document.getElementById('zonesCount').textContent = count;
        }

        function generateGUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        function showStatus(message, type, duration = 4000) {
            // Создаем элемент статуса если его нет
            let statusElement = document.getElementById('statusMessage');
            if (!statusElement) {
                statusElement = document.createElement('div');
                statusElement.id = 'statusMessage';
                statusElement.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 12px 16px;
                    border-radius: 6px;
                    color: white;
                    font-weight: bold;
                    z-index: 1000;
                    max-width: 350px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                `;
                document.body.appendChild(statusElement);
            }

            // Устанавливаем цвет в зависимости от типа
            const colors = {
                'success': '#38a169',
                'error': '#e53e3e', 
                'warning': '#d69e2e',
                'info': '#3182ce'
            };

            statusElement.style.backgroundColor = colors[type] || colors.info;
            statusElement.textContent = message;
            statusElement.style.display = 'block';
            statusElement.style.transform = 'translateX(0)';

            // Консольный лог для отладки
            const emoji = {
                'success': '✅',
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️'
            };
            console.log(`${emoji[type] || 'ℹ️'} ${message}`);

            // Автоматически скрываем
            setTimeout(() => {
                statusElement.style.transform = 'translateX(100%)';
                setTimeout(() => {
                    statusElement.style.display = 'none';
                }, 300);
            }, duration);
        }

        // Инициализация при загрузке страницы
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 Инициализация обучающего модуля');

            // Инициализируем DOM элементы
            uploadArea = document.getElementById('uploadArea');
            fileInput = document.getElementById('fileInput');
            progressContainer = document.getElementById('progressContainer');
            progressBar = document.getElementById('progressBar');
            progressText = document.getElementById('progressText');
            progressDetails = document.getElementById('progressDetails');
            fileList = document.getElementById('fileList');
            videoPanel = document.getElementById('videoPanel');
            frameSlider = document.getElementById('frameSlider');
            frameInfo = document.getElementById('frameInfo');
            videoCanvas = document.getElementById('videoCanvas');
            overlayCanvas = document.getElementById('overlayCanvas');

            // Инициализируем контексты canvas
            if (videoCanvas && overlayCanvas) {
                ctx = videoCanvas.getContext('2d');
                overlayCtx = overlayCanvas.getContext('2d');
                console.log('✅ Canvas контексты инициализированы');
            } else {
                console.error('❌ Canvas элементы не найдены');
            }

            // Проверяем наличие ключевых элементов
            const requiredElements = [
                {id: 'uploadArea', element: uploadArea},
                {id: 'fileInput', element: fileInput},
                {id: 'fileList', element: fileList},
                {id: 'totalFiles', element: document.getElementById('totalFiles')},
                {id: 'totalSize', element: document.getElementById('totalSize')},
                {id: 'videoPanel', element: videoPanel},
                {id: 'videoCanvas', element: videoCanvas},
                {id: 'overlayCanvas', element: overlayCanvas}
            ];

            requiredElements.forEach(({id, element}) => {
                if (element) {
                    console.log(`✅ Элемент найден: ${id}`);
                } else {
                    console.error(`❌ Элемент НЕ найден: ${id}`);
                }
            });

            // Инициализируем обработчики событий
            setupEventListeners();

            // Загружаем список файлов
            console.log('📂 Запуск загрузки файлов...');
            setTimeout(() => {
                loadFileList();
            }, 500); // Небольшая задержка для полной загрузки DOM

            // Инициализируем счетчики
            updateZonesCount();
            updateAnnotatedCount();

            // Добавляем отладочную функцию
            window.debugTraining = function() {
                console.log('=== DEBUG TRAINING ===');
                console.log('Current camera:', currentCamera);
                console.log('Zones:', zones);
                console.log('Detected objects:', detectedObjects);
                console.log('Current frame:', currentFrame);
                console.log('Total frames:', totalFrames);
                console.log('Selected object:', selectedObject);
                console.log('Video panel active:', videoPanel ? videoPanel.classList.contains('active') : 'N/A');
                console.log('FileList element:', fileList);
                console.log('TotalFiles element:', document.getElementById('totalFiles'));
            };

            console.log('💡 Для отладки используйте: window.debugTraining()');
            console.log('🎉 Инициализация завершена');
        });

        // Обновляем счетчики каждые 30 секунд
        setInterval(updateAnnotatedCount, 30000);
    </script>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Запуск модуля"""
        self.app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    module = AdvancedTrainingModule()
    module.run(debug=True)