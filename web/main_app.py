# web/main_app.py
"""Главный веб-интерфейс системы подсчета хлеба"""

from core.imports import *
from core.tpu_manager import TPUManager
from werkzeug.utils import secure_filename
import psutil
import time


class ProductionMonitorApp:
    """Главное веб-приложение"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.config['SECRET_KEY'] = 'bread_counter_2025'

        # Менеджеры
        self.tpu_manager = TPUManager()
        self.production_jobs = {}  # {job_id: job_info}
        self.cameras_config = self._load_cameras_config()

        self._setup_routes()

    def _load_cameras_config(self):
        """Загрузка конфигурации камер"""
        try:
            config_path = 'config/cameras.yaml'
            if os.path.exists(config_path) and YAML_AVAILABLE:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                    if config:
                        print(f"📖 Загружена конфигурация из {config_path}")
                        print(f"   Камер: {len(config.get('cameras', []))}")
                        print(f"   Система: {bool(config.get('system'))}")
                        print(f"   Классы: {len(config.get('classes', []))}")
                        return config
                    else:
                        print(f"⚠️  Файл {config_path} пустой, создаем новую конфигурацию")
            else:
                print(f"📝 Файл {config_path} не найден, создаем базовую конфигурацию")

            # Создаем базовую конфигурацию
            default_config = {
                'cameras': [],
                'system': {
                    'tpu_devices': 1,
                    'frame_rate': 15,
                    'detection_threshold': 0.5,
                    'tracking_max_distance': 100
                },
                'classes': [
                    {'name': 'bread', 'color': [0, 255, 0]},
                    {'name': 'bun', 'color': [255, 0, 0]},
                    {'name': 'loaf', 'color': [0, 0, 255]},
                    {'name': 'pastry', 'color': [255, 255, 0]}
                ]
            }
            self._save_cameras_config(default_config)
            return default_config

        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации камер: {e}")
            return {'cameras': [], 'system': {}, 'classes': []}

    def _save_cameras_config(self, config):
        """Сохранение конфигурации камер в YAML"""
        try:
            os.makedirs('config', exist_ok=True)
            config_path = 'config/cameras.yaml'

            # Логируем что сохраняем
            print(f"💾 Сохранение конфигурации в {config_path}")
            print(f"   Камер: {len(config.get('cameras', []))}")
            print(f"   Система: {bool(config.get('system'))}")
            print(f"   Классы: {len(config.get('classes', []))}")

            if YAML_AVAILABLE:
                with open(config_path, 'w', encoding='utf-8') as f:
                    # Правильные параметры для PyYAML
                    yaml.dump(config, f,
                              default_flow_style=False,
                              allow_unicode=True,
                              indent=2,
                              sort_keys=False)
                print(f"✅ Конфигурация сохранена в {config_path}")
                return True
            else:
                # Fallback to JSON if YAML not available
                print("⚠️  YAML недоступен, сохраняем в JSON")
                import json
                with open(config_path.replace('.yaml', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                return True
        except Exception as e:
            print(f"❌ Ошибка сохранения конфигурации: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_template())

        @self.app.route('/api/system/status')
        def system_status():
            """Статус системы"""
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            tpu_status = {
                'available': self.tpu_manager.is_available(),
                'device_count': self.tpu_manager.get_device_count(),
                'devices': self.tpu_manager.list_devices()
            }

            return jsonify({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3),
                'tpu': tpu_status,
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/cameras')
        def get_cameras():
            """Получить список камер"""
            try:
                config = self._load_cameras_config()
                print(f"📡 API запрос конфигурации камер - возвращаем {len(config.get('cameras', []))} камер")
                return jsonify(config)
            except Exception as e:
                print(f"❌ Ошибка API получения камер: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/cameras', methods=['POST'])
        def update_cameras():
            """Обновить конфигурацию камер"""
            try:
                new_config = request.json
                print(f"🔄 Получен запрос на обновление конфигурации камер")

                # Валидация данных
                if 'cameras' not in new_config:
                    return jsonify({'status': 'error', 'message': 'Отсутствует список камер'}), 400

                # Проверяем каждую камеру
                for i, camera in enumerate(new_config['cameras']):
                    required_fields = ['camera_ip', 'login', 'password', 'oven_name']
                    for field in required_fields:
                        if field not in camera or not camera[field]:
                            return jsonify({
                                'status': 'error',
                                'message': f'Камера {i + 1}: отсутствует поле {field}'
                            }), 400

                    # Проверка формата IP
                    ip = camera['camera_ip']
                    ip_parts = ip.split('.')
                    if len(ip_parts) != 4 or not all(part.isdigit() and 0 <= int(part) <= 255 for part in ip_parts):
                        return jsonify({
                            'status': 'error',
                            'message': f'Некорректный IP адрес: {ip}'
                        }), 400

                print(f"✅ Валидация прошла успешно, камер: {len(new_config['cameras'])}")

                # Сохраняем конфигурацию
                success = self._save_cameras_config(new_config)

                if success:
                    self.cameras_config = new_config
                    return jsonify({'status': 'success', 'message': 'Конфигурация сохранена'})
                else:
                    return jsonify({'status': 'error', 'message': 'Ошибка сохранения файла'}), 500

            except Exception as e:
                print(f"❌ Ошибка API обновления камер: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/api/cameras/<int:camera_id>/test', methods=['POST'])
        def test_camera(camera_id):
            """Тестирование подключения к камере"""
            try:
                config = self._load_cameras_config()
                cameras = config.get('cameras', [])

                camera = None
                for cam in cameras:
                    if cam.get('oven_id') == camera_id:
                        camera = cam
                        break

                if not camera:
                    return jsonify({'status': 'error', 'message': 'Камера не найдена'}), 404

                # Формируем RTSP URL
                rtsp_url = f"rtsp://{camera['login']}:{camera['password']}@{camera['camera_ip']}/stream1"

                # Тестируем подключение
                import cv2
                cap = cv2.VideoCapture(rtsp_url)

                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        return jsonify({
                            'status': 'success',
                            'message': f'Камера {camera["oven_name"]} доступна',
                            'resolution': f'{frame.shape[1]}x{frame.shape[0]}' if ret else 'Неизвестно'
                        })
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': 'Камера подключена, но не передает видео'
                        })
                else:
                    cap.release()
                    return jsonify({
                        'status': 'error',
                        'message': 'Не удалось подключиться к камере. Проверьте IP, логин и пароль'
                    })

            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Ошибка тестирования: {str(e)}'}), 500

        @self.app.route('/api/cameras/debug')
        def debug_cameras_config():
            """Отладочная информация о конфигурации камер"""
            try:
                config_path = 'config/cameras.yaml'

                debug_info = {
                    'config_path': config_path,
                    'file_exists': os.path.exists(config_path),
                    'yaml_available': YAML_AVAILABLE
                }

                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    debug_info['file_content'] = file_content
                    debug_info['file_size'] = len(file_content)

                    if YAML_AVAILABLE:
                        try:
                            parsed_config = yaml.safe_load(file_content)
                            debug_info['parsed_config'] = parsed_config
                            debug_info['cameras_count'] = len(parsed_config.get('cameras', []))
                        except Exception as e:
                            debug_info['yaml_parse_error'] = str(e)
                else:
                    debug_info['message'] = 'Файл конфигурации не существует'

                return jsonify(debug_info)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/jobs')
        def get_jobs():
            """Получить текущие задания"""
            return jsonify(self.production_jobs)

        @self.app.route('/api/jobs/start', methods=['POST'])
        def start_job():
            """Запуск задания"""
            data = request.json
            job_id = str(uuid.uuid4())

            self.production_jobs[job_id] = {
                'id': job_id,
                'camera_id': data.get('camera_id'),
                'product_type': data.get('product_type', 'bread'),
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'count': 0,
                'fps': 0,
                'last_detection': None
            }

            # Здесь запуск реального процесса детекции
            self._start_detection_process(job_id)

            return jsonify({'status': 'success', 'job_id': job_id})

        @self.app.route('/api/jobs/<job_id>/stop', methods=['POST'])
        def stop_job(job_id):
            """Остановка задания"""
            if job_id in self.production_jobs:
                self.production_jobs[job_id]['status'] = 'stopped'
                self.production_jobs[job_id]['stopped_at'] = datetime.now().isoformat()

                # Здесь остановка процесса детекции
                self._stop_detection_process(job_id)

                return jsonify({'status': 'success'})

            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        @self.app.route('/api/jobs/<job_id>/restart', methods=['POST'])
        def restart_job(job_id):
            """Перезапуск задания"""
            if job_id in self.production_jobs:
                job = self.production_jobs[job_id]
                job['status'] = 'running'
                job['restarted_at'] = datetime.now().isoformat()
                job['count'] = 0

                self._restart_detection_process(job_id)

                return jsonify({'status': 'success'})

            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        @self.app.route('/training')
        def training_interface():
            """Обучающий интерфейс"""
            # Получаем текущий хост
            host = request.host.split(':')[0]  # Убираем порт если есть
            training_url = f"http://{host}:5001/training"

            return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Перенаправление...</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 2rem; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .loading {{ display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #3182ce; border-radius: 50%; animation: spin 1s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .btn {{ padding: 0.75rem 1.5rem; background: #3182ce; color: white; text-decoration: none; border-radius: 6px; display: inline-block; margin: 1rem; }}
        .error {{ color: #e53e3e; margin-top: 1rem; }}
    </style>
    <script>
        let countdown = 3;
        let trainingUrl = '{training_url}';

        function updateCounter() {{
            document.getElementById('counter').textContent = countdown;
            if (countdown <= 0) {{
                window.location.href = trainingUrl;
            }} else {{
                countdown--;
                setTimeout(updateCounter, 1000);
            }}
        }}

        function checkTrainingModule() {{
            fetch(trainingUrl.replace('/training', '/api/training/files'))
                .then(response => {{
                    if (response.ok) {{
                        // Модуль обучения доступен
                        document.getElementById('status').innerHTML = `
                            <div class="loading"></div>
                            <p>Перенаправление на модуль обучения через <span id="counter">3</span> сек...</p>
                        `;
                        updateCounter();
                    }} else {{
                        throw new Error('Модуль недоступен');
                    }}
                }})
                .catch(error => {{
                    document.getElementById('status').innerHTML = `
                        <div class="error">❌ Модуль обучения недоступен на {training_url}</div>
                        <p>Возможные причины:</p>
                        <ul style="text-align: left; display: inline-block;">
                            <li>Модуль обучения не запущен</li>
                            <li>Порт 5001 заблокирован</li>
                            <li>Проблемы с сетью</li>
                        </ul>
                        <p><strong>Решение:</strong> Убедитесь что <code>python main.py</code> запустил оба модуля</p>
                    `;
                }});
        }}

        window.onload = checkTrainingModule;
    </script>
</head>
<body>
    <div class="container">
        <h1>🧠 Переход к модулю обучения</h1>
        <div id="status">
            <div class="loading"></div>
            <p>Проверка доступности модуля обучения...</p>
        </div>
        <div style="margin-top: 2rem;">
            <a href="{training_url}" class="btn">🔗 Перейти принудительно</a>
            <a href="/" class="btn" style="background: #718096;">← Вернуться на главную</a>
        </div>
    </div>
</body>
</html>
            '''

        @self.app.route('/api/training/files')
        def list_training_files():
            """Список загруженных файлов для обучения"""
            try:
                upload_dir = 'uploads'
                files = []

                if os.path.exists(upload_dir):
                    for filename in os.listdir(upload_dir):
                        filepath = os.path.join(upload_dir, filename)
                        if os.path.isfile(filepath):
                            stat = os.stat(filepath)
                            files.append({
                                'name': filename,
                                'size_bytes': stat.st_size,
                                'size_gb': stat.st_size / (1024 ** 3),
                                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'extension': os.path.splitext(filename)[1].lower()
                            })

                # Сортируем по дате изменения
                files.sort(key=lambda x: x['modified'], reverse=True)
                return jsonify({'files': files})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/files/<filename>', methods=['DELETE'])
        def delete_training_file(filename):
            """Удаление файла"""
            try:
                filepath = os.path.join('uploads', secure_filename(filename))
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return jsonify({'status': 'success'})
                else:
                    return jsonify({'error': 'Файл не найден'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/training/files/<filename>/rename', methods=['POST'])
        def rename_training_file(filename):
            """Переименование файла"""
            try:
                data = request.json
                new_name = secure_filename(data.get('new_name', ''))

                old_path = os.path.join('uploads', secure_filename(filename))
                new_path = os.path.join('uploads', new_name)

                if os.path.exists(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    return jsonify({'status': 'success', 'new_name': new_name})
                else:
                    return jsonify({'error': 'Ошибка переименования'}), 400

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def _start_detection_process(self, job_id):
        """Запуск процесса детекции (заглушка)"""
        # Здесь будет реальная логика запуска детекции
        pass

    def _stop_detection_process(self, job_id):
        """Остановка процесса детекции (заглушка)"""
        pass

    def _restart_detection_process(self, job_id):
        """Перезапуск процесса детекции (заглушка)"""
        self._stop_detection_process(job_id)
        self._start_detection_process(job_id)

    def _get_dashboard_template(self):
        """HTML шаблон главной панели"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🥖 Система подсчета хлеба</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; }
        .header { background: #2d3748; color: white; padding: 1rem; }
        .container { max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h3 { margin-bottom: 1rem; color: #2d3748; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .status-item { background: #f7fafc; padding: 1rem; border-radius: 6px; text-align: center; }
        .status-value { font-size: 2rem; font-weight: bold; color: #2b6cb0; }
        .btn { padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #3182ce; color: white; }
        .btn-success { background: #38a169; color: white; }
        .btn-danger { background: #e53e3e; color: white; }
        .btn-secondary { background: #718096; color: white; }
        .job-item { background: #f7fafc; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }
        .job-controls { margin-top: 0.5rem; }
        .job-controls button { margin-right: 0.5rem; }
        .camera-form { margin-bottom: 1rem; }
        .camera-form input, .camera-form select { 
            width: 100%; padding: 0.5rem; margin-bottom: 0.5rem; border: 1px solid #e2e8f0; border-radius: 4px; 
        }
        .camera-form input[type="password"] {
            background: #f7fafc;
        }
        .btn-sm { padding: 0.25rem 0.5rem; font-size: 12px; margin: 0 0.125rem; }
        .nav-link { 
            display: inline-block; padding: 1rem 2rem; background: #4299e1; color: white; 
            text-decoration: none; border-radius: 6px; margin-right: 1rem; 
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🥖 Система подсчета хлеба</h1>
        <div style="margin-top: 1rem;">
                                <a href="/training" class="nav-link">🧠 Обучение модели</a>
                    <a href="/api/system/status" target="_blank" class="nav-link">📊 API статус</a>
            <a href="#" onclick="refreshData()" class="nav-link">🔄 Обновить</a>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <!-- Системный статус -->
            <div class="card">
                <h3>📊 Системный статус</h3>
                <div class="status-grid" id="systemStatus">
                    <div class="status-item">
                        <div class="status-value" id="cpuValue">-</div>
                        <div>CPU %</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value" id="memoryValue">-</div>
                        <div>RAM %</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value" id="tpuValue">-</div>
                        <div>TPU устройств</div>
                    </div>
                </div>
            </div>

            <!-- Камеры/Печи -->
            <div class="card">
                <h3>📹 Настройка камер</h3>
                <div class="camera-form">
                    <input type="text" id="cameraIP" placeholder="IP камеры (192.168.1.100)">
                    <input type="text" id="cameraLogin" placeholder="Логин (admin)" value="admin">
                    <input type="password" id="cameraPassword" placeholder="Пароль">
                    <input type="text" id="ovenName" placeholder="Название печи">
                    <input type="text" id="workshopName" placeholder="Название цеха">
                    <select id="productType">
                        <option value="bread">Хлеб</option>
                        <option value="bun">Булочки</option>
                        <option value="loaf">Батон</option>
                        <option value="pastry">Выпечка</option>
                    </select>
                    <button class="btn btn-primary" onclick="addCamera()">➕ Добавить камеру</button>
                    <button class="btn btn-secondary" onclick="debugConfig()" style="font-size: 12px;">🔍 Debug</button>
                </div>
                <div id="camerasList"></div>
            </div>
        </div>

        <!-- Текущие задания -->
        <div class="card" style="margin-top: 2rem;">
            <h3>🎯 Текущие задания</h3>
            <button class="btn btn-success" onclick="startNewJob()">▶️ Начать новое задание</button>
            <div id="jobsList" style="margin-top: 1rem;"></div>
        </div>
    </div>

    <script>
        let cameras = [];
        let jobs = {};
        let systemConfig = {};
        let classesConfig = [];

        function refreshData() {
            loadSystemStatus();
            loadCameras();
            loadJobs();
        }

        function loadSystemStatus() {
            fetch('/api/system/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cpuValue').textContent = data.cpu_percent.toFixed(1);
                    document.getElementById('memoryValue').textContent = data.memory_percent.toFixed(1);
                    document.getElementById('tpuValue').textContent = data.tpu.device_count;
                });
        }

        function loadCameras() {
            fetch('/api/cameras')
                .then(response => response.json())
                .then(data => {
                    // Сохраняем полную конфигурацию
                    cameras = data.cameras || [];
                    systemConfig = data.system || {
                        tpu_devices: 1,
                        frame_rate: 15,
                        detection_threshold: 0.5,
                        tracking_max_distance: 100
                    };
                    classesConfig = data.classes || [
                        {name: "bread", color: [0, 255, 0]},
                        {name: "bun", color: [255, 0, 0]},
                        {name: "loaf", color: [0, 0, 255]},
                        {name: "pastry", color: [255, 255, 0]}
                    ];

                    renderCameras();
                    console.log('Загружена конфигурация:', { cameras: cameras.length, system: systemConfig, classes: classesConfig.length });
                })
                .catch(error => {
                    console.error('Ошибка загрузки камер:', error);
                    showStatus('Ошибка загрузки списка камер', 'error');
                });
        }

        function renderCameras() {
            const list = document.getElementById('camerasList');
            list.innerHTML = cameras.map((camera, index) => 
                `<div class="job-item">
                    <div><strong>${camera.oven_name}</strong> - ${camera.camera_ip}</div>
                    <div style="font-size: 12px; color: #666;">
                        Цех: ${camera.workshop_name || 'Не указан'} | 
                        Логин: ${camera.login} | 
                        Продукт: ${camera.product_type || 'bread'}
                    </div>
                    <div class="job-controls">
                        <button class="btn btn-secondary btn-sm" onclick="testCamera(${index})">🔧 Тест</button>
                        <button class="btn btn-warning btn-sm" onclick="editCamera(${index})">✏️ Изменить</button>
                        <button class="btn btn-danger btn-sm" onclick="removeCamera(${index})">🗑️ Удалить</button>
                    </div>
                </div>`
            ).join('');
        }

        function addCamera() {
            const ip = document.getElementById('cameraIP').value.trim();
            const login = document.getElementById('cameraLogin').value.trim();
            const password = document.getElementById('cameraPassword').value.trim();
            const name = document.getElementById('ovenName').value.trim();
            const workshop = document.getElementById('workshopName').value.trim();
            const type = document.getElementById('productType').value;

            // Валидация
            if (!ip || !login || !password || !name) {
                alert('Заполните все обязательные поля: IP, логин, пароль, название печи');
                return;
            }

            // Проверка формата IP
            const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$/;
            if (!ipRegex.test(ip)) {
                alert('Введите корректный IP адрес (например: 192.168.1.100)');
                return;
            }

            // Проверка на дублирование IP
            if (cameras.some(camera => camera.camera_ip === ip)) {
                alert('Камера с таким IP уже существует');
                return;
            }

            const newCamera = {
                oven_id: cameras.length + 1,
                camera_ip: ip,
                login: login,
                password: password,
                oven_name: name,
                workshop_name: workshop,
                enterprise_name: "Хлебозавод",
                product_type: type
            };

            cameras.push(newCamera);

            // Используем существующую конфигурацию
            const yamlData = {
                cameras: cameras,
                system: systemConfig,
                classes: classesConfig
            };

            fetch('/api/cameras', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(yamlData)
            }).then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Очищаем форму
                    document.getElementById('cameraIP').value = '';
                    document.getElementById('cameraPassword').value = '';
                    document.getElementById('ovenName').value = '';
                    document.getElementById('workshopName').value = '';

                    renderCameras();
                    showStatus('Камера добавлена успешно', 'success');
                } else {
                    showStatus('Ошибка сохранения: ' + data.message, 'error');
                }
            }).catch(error => {
                showStatus('Ошибка сети: ' + error.message, 'error');
            });
        }

        function testCamera(index) {
            const camera = cameras[index];
            showStatus(`Тестирование камеры ${camera.oven_name}...`, 'info');

            fetch(`/api/cameras/${camera.oven_id}/test`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ ${data.message}`, 'success');
                } else {
                    showStatus(`❌ ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Ошибка тестирования: ${error.message}`, 'error');
            });
        }

        function editCamera(index) {
            const camera = cameras[index];

            // Заполняем форму данными камеры
            document.getElementById('cameraIP').value = camera.camera_ip;
            document.getElementById('cameraLogin').value = camera.login;
            document.getElementById('cameraPassword').value = camera.password;
            document.getElementById('ovenName').value = camera.oven_name;
            document.getElementById('workshopName').value = camera.workshop_name || '';
            document.getElementById('productType').value = camera.product_type || 'bread';

            // Удаляем старую камеру
            removeCamera(index, false);

            showStatus('Данные камеры загружены в форму для редактирования', 'info');
        }

        function removeCamera(index, showMessage = true) {
            if (showMessage && !confirm(`Удалить камеру "${cameras[index].oven_name}"?`)) {
                return;
            }

            cameras.splice(index, 1);

            // Обновляем oven_id для остальных камер
            cameras.forEach((camera, i) => {
                camera.oven_id = i + 1;
            });

            // Используем существующую конфигурацию
            const yamlData = {
                cameras: cameras,
                system: systemConfig,
                classes: classesConfig
            };

            fetch('/api/cameras', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(yamlData)
            }).then(() => {
                renderCameras();
                if (showMessage) {
                    showStatus('Камера удалена', 'success');
                }
            });
        }

        function showStatus(message, type) {
            // Создаем элемент статуса если его нет
            let statusElement = document.getElementById('statusMessage');
            if (!statusElement) {
                statusElement = document.createElement('div');
                statusElement.id = 'statusMessage';
                statusElement.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 10px 15px;
                    border-radius: 5px;
                    color: white;
                    font-weight: bold;
                    z-index: 1000;
                    max-width: 300px;
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

            // Автоматически скрываем через 4 секунды
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 4000);
        }

        function debugConfig() {
            fetch('/api/cameras/debug')
                .then(response => response.json())
                .then(data => {
                    console.log('Debug конфигурации камер:', data);

                    let message = `Отладка конфигурации:\n`;
                    message += `Файл: ${data.config_path}\n`;
                    message += `Существует: ${data.file_exists}\n`;
                    message += `YAML доступен: ${data.yaml_available}\n`;

                    if (data.file_exists) {
                        message += `Размер файла: ${data.file_size} байт\n`;
                        if (data.cameras_count !== undefined) {
                            message += `Камер в файле: ${data.cameras_count}\n`;
                        }
                        if (data.yaml_parse_error) {
                            message += `Ошибка парсинга: ${data.yaml_parse_error}\n`;
                        }
                    }

                    alert(message);

                    // Также выводим в консоль для подробного анализа
                    if (data.file_content) {
                        console.log('Содержимое файла:', data.file_content);
                    }
                    if (data.parsed_config) {
                        console.log('Разобранная конфигурация:', data.parsed_config);
                    }
                })
                .catch(error => {
                    console.error('Ошибка debug:', error);
                    showStatus('Ошибка получения debug информации', 'error');
                });
        }

        function loadJobs() {
            fetch('/api/jobs')
                .then(response => response.json())
                .then(data => {
                    jobs = data;
                    renderJobs();
                });
        }

        function renderJobs() {
            const list = document.getElementById('jobsList');
            const jobEntries = Object.entries(jobs);

            if (jobEntries.length === 0) {
                list.innerHTML = '<p>Нет активных заданий</p>';
                return;
            }

            list.innerHTML = jobEntries.map(([jobId, job]) => 
                `<div class="job-item">
                    <div><strong>Задание:</strong> ${job.product_type} (Камера ${job.camera_id})</div>
                    <div><strong>Статус:</strong> ${job.status}</div>
                    <div><strong>Подсчет:</strong> ${job.count}</div>
                    <div><strong>FPS:</strong> ${job.fps}</div>
                    <div class="job-controls">
                        ${job.status === 'running' ? 
                            `<button class="btn btn-danger" onclick="stopJob('${jobId}')">⏹️ Стоп</button>` :
                            `<button class="btn btn-success" onclick="restartJob('${jobId}')">🔄 Перезапуск</button>`
                        }
                    </div>
                </div>`
            ).join('');
        }

        function startNewJob() {
            if (cameras.length === 0) {
                alert('Сначала добавьте камеру');
                return;
            }

            const cameraId = cameras[0].oven_id;
            const productType = cameras[0].product_type || 'bread';

            fetch('/api/jobs/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera_id: cameraId, product_type: productType})
            }).then(() => loadJobs());
        }

        function stopJob(jobId) {
            fetch(`/api/jobs/${jobId}/stop`, {method: 'POST'})
                .then(() => loadJobs());
        }

        function restartJob(jobId) {
            fetch(`/api/jobs/${jobId}/restart`, {method: 'POST'})
                .then(() => loadJobs());
        }

        // Автообновление каждые 5 секунд
        setInterval(refreshData, 5000);

        // Первоначальная загрузка
        refreshData();
    </script>
</body>
</html>
        '''

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Запуск приложения"""
        self.app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    app = ProductionMonitorApp()
    app.run(debug=True)