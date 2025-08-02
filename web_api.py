# web_api.py - REST API для мониторинга системы
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import os


class MonitoringAPI:
    """REST API для мониторинга многокамерной системы"""

    def __init__(self, camera_manager):
        self.app = Flask(__name__)
        CORS(self.app)
        self.camera_manager = camera_manager

        # Настройка маршрутов
        self._setup_routes()

        print("🌐 Web API инициализирован")

    def _setup_routes(self):
        """Настройка маршрутов API"""

        @self.app.route('/')
        def dashboard():
            """Главная страница мониторинга"""
            return render_template_string(self._get_dashboard_template())

        @self.app.route('/api/overview')
        def system_overview():
            """Общий обзор системы"""
            return jsonify(self.camera_manager.get_system_overview())

        @self.app.route('/api/oven/<int:oven_id>')
        def oven_status(oven_id):
            """Статус конкретной печи"""
            return jsonify(self.camera_manager.get_oven_status(oven_id))

        @self.app.route('/api/ovens')
        def all_ovens():
            """Статус всех печей"""
            ovens = {}
            for oven_id in self.camera_manager.cameras.keys():
                ovens[oven_id] = self.camera_manager.get_oven_status(oven_id)
            return jsonify(ovens)

        @self.app.route('/api/stats/hourly')
        def hourly_stats():
            """Почасовая статистика производства"""
            # В реальной системе здесь будет запрос к БД
            return jsonify(self._get_hourly_stats())

        @self.app.route('/api/alerts')
        def system_alerts():
            """Системные уведомления"""
            return jsonify(self._get_system_alerts())

        @self.app.route('/api/batches/active')
        def active_batches():
            """Активные партии производства"""
            batches = []
            for oven_id, counter in self.camera_manager.counters.items():
                if counter.current_batch:
                    batch_info = {
                        'oven_id': oven_id,
                        'product_name': counter.current_product.name if counter.current_product else 'Unknown',
                        'start_time': counter.current_batch.start_time.isoformat(),
                        'count': counter.current_batch.total_count,
                        'defects': counter.current_batch.defect_count,
                        'duration_minutes': (datetime.now() - counter.current_batch.start_time).total_seconds() / 60
                    }
                    batches.append(batch_info)

            return jsonify(batches)

        @self.app.route('/api/performance')
        def performance_stats():
            """Статистика производительности системы"""
            tpu_stats = self.camera_manager.tpu_pool.get_stats()

            # Собираем статистику по камерам
            camera_stats = []
            for oven_id, stats in self.camera_manager.stats.items():
                camera_stats.append({
                    'oven_id': oven_id,
                    'fps': stats['current_fps'],
                    'frames_processed': stats['frames_processed'],
                    'detections': stats['detections_count'],
                    'last_activity': stats['last_activity']
                })

            return jsonify({
                'tpu_pool': tpu_stats,
                'cameras': camera_stats,
                'system_uptime': self._get_system_uptime()
            })

        @self.app.route('/api/config')
        def system_config():
            """Конфигурация системы"""
            config_info = {
                'total_cameras': len(self.camera_manager.cameras),
                'tpu_devices': self.camera_manager.tpu_pool.num_devices,
                'ovens': []
            }

            for oven_id, camera in self.camera_manager.cameras.items():
                config_info['ovens'].append({
                    'oven_id': oven_id,
                    'camera_ip': camera.camera_ip,
                    'status': 'active' if oven_id in self.camera_manager.stats else 'inactive'
                })

            return jsonify(config_info)

    def _get_dashboard_template(self):
        """HTML шаблон для dashboard"""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🥖 Система подсчета хлеба - Мониторинг</title>
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

        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #27ae60;
            margin-bottom: 10px;
        }

        .ovens-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .oven-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .oven-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .oven-title {
            font-size: 1.4em;
            color: #2c3e50;
            font-weight: bold;
        }

        .status-indicator {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #27ae60;
            animation: pulse 2s infinite;
        }

        .status-indicator.inactive {
            background: #e74c3c;
            animation: none;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .oven-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .oven-stat {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .oven-stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }

        .oven-stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }

        .batch-info {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }

        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #3498db;
            color: white;
            border: none;
            padding: 15px;
            border-radius: 50%;
            font-size: 1.2em;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }

        .refresh-btn:hover {
            background: #2980b9;
            transform: scale(1.1);
        }

        .last-update {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-style: italic;
        }

        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🥖 Система подсчета хлеба</h1>
            <p>Мониторинг производства в реальном времени</p>
        </div>

        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <h3>📹 Активные камеры</h3>
                <div class="stat-value" id="activeCameras">-</div>
                <p>из <span id="totalCameras">-</span> всего</p>
            </div>

            <div class="stat-card">
                <h3>🔥 Активные партии</h3>
                <div class="stat-value" id="activeBatches">-</div>
                <p>производство</p>
            </div>

            <div class="stat-card">
                <h3>🧠 TPU устройства</h3>
                <div class="stat-value" id="tpuDevices">-</div>
                <p>ускорителей</p>
            </div>

            <div class="stat-card">
                <h3>📊 Кадров/сек</h3>
                <div class="stat-value" id="totalFps">-</div>
                <p>общая производительность</p>
            </div>
        </div>

        <div class="ovens-grid" id="ovensGrid">
            <!-- Карточки печей загружаются динамически -->
        </div>

        <div class="last-update" id="lastUpdate">
            Загрузка данных...
        </div>
    </div>

    <button class="refresh-btn" onclick="loadData()" title="Обновить данные">
        🔄
    </button>

    <script>
        let isLoading = false;

        async function loadData() {
            if (isLoading) return;

            isLoading = true;
            document.body.classList.add('loading');

            try {
                // Загружаем общую статистику
                const overview = await fetch('/api/overview').then(r => r.json());
                const ovens = await fetch('/api/ovens').then(r => r.json());
                const batches = await fetch('/api/batches/active').then(r => r.json());
                const performance = await fetch('/api/performance').then(r => r.json());

                // Обновляем общую статистику
                document.getElementById('activeCameras').textContent = overview.active_cameras || 0;
                document.getElementById('totalCameras').textContent = overview.total_cameras || 0;
                document.getElementById('activeBatches').textContent = batches.length || 0;
                document.getElementById('tpuDevices').textContent = performance.tpu_pool?.active_devices || 0;

                // Считаем общий FPS
                const totalFps = Object.values(ovens).reduce((sum, oven) => sum + (oven.fps || 0), 0);
                document.getElementById('totalFps').textContent = totalFps.toFixed(1);

                // Обновляем карточки печей
                updateOvensGrid(ovens, batches);

                // Время последнего обновления
                document.getElementById('lastUpdate').textContent = 
                    `Последнее обновление: ${new Date().toLocaleTimeString()}`;

            } catch (error) {
                console.error('Ошибка загрузки данных:', error);
                document.getElementById('lastUpdate').textContent = 
                    `Ошибка загрузки: ${error.message}`;
            } finally {
                isLoading = false;
                document.body.classList.remove('loading');
            }
        }

        function updateOvensGrid(ovens, batches) {
            const grid = document.getElementById('ovensGrid');
            grid.innerHTML = '';

            // Создаем словарь активных партий
            const activeBatchesMap = {};
            batches.forEach(batch => {
                activeBatchesMap[batch.oven_id] = batch;
            });

            // Создаем карточки печей
            Object.entries(ovens).forEach(([ovenId, oven]) => {
                const batch = activeBatchesMap[ovenId];
                const isActive = oven.fps > 0 && Date.now() - oven.last_activity * 1000 < 60000;

                const card = document.createElement('div');
                card.className = 'oven-card';
                card.innerHTML = `
                    <div class="oven-header">
                        <div class="oven-title">🔥 Печь ${ovenId}</div>
                        <div class="status-indicator ${isActive ? '' : 'inactive'}"></div>
                    </div>

                    <div class="oven-stats">
                        <div class="oven-stat">
                            <div class="oven-stat-label">FPS</div>
                            <div class="oven-stat-value">${oven.fps || 0}</div>
                        </div>
                        <div class="oven-stat">
                            <div class="oven-stat-label">Кадров</div>
                            <div class="oven-stat-value">${oven.frames_processed || 0}</div>
                        </div>
                        <div class="oven-stat">
                            <div class="oven-stat-label">Детекций</div>
                            <div class="oven-stat-value">${oven.detections_count || 0}</div>
                        </div>
                        <div class="oven-stat">
                            <div class="oven-stat-label">Объектов</div>
                            <div class="oven-stat-value">${oven.tracked_objects || 0}</div>
                        </div>
                    </div>

                    ${batch ? `
                        <div class="batch-info">
                            <strong>🥖 Активная партия:</strong><br>
                            ${batch.product_name}<br>
                            <small>Подсчет: ${batch.count} шт, Брак: ${batch.defects} шт</small><br>
                            <small>Время: ${Math.round(batch.duration_minutes)} мин</small>
                        </div>
                    ` : '<div style="text-align: center; color: #666; margin-top: 15px;">⏸️ Партия не активна</div>'}
                `;

                grid.appendChild(card);
            });
        }

        // Автоматическое обновление каждые 5 секунд
        setInterval(loadData, 5000);

        // Первоначальная загрузка
        loadData();
    </script>
</body>
</html>
        '''

    def _get_hourly_stats(self):
        """Получение почасовой статистики (заглушка)"""
        # В реальной системе здесь будет запрос к БД
        now = datetime.now()
        stats = []

        for i in range(24):
            hour_time = now - timedelta(hours=i)
            stats.append({
                'hour': hour_time.strftime('%H:00'),
                'total_count': 45 + (i * 3),  # Заглушка
                'defects': 2 + (i % 3),  # Заглушка
                'efficiency': 95.5 + (i % 5)  # Заглушка
            })

        return stats[::-1]  # Разворачиваем чтобы начинать с утра

    def _get_system_alerts(self):
        """Получение системных уведомлений"""
        alerts = []

        # Проверяем активность камер
        for oven_id, stats in self.camera_manager.stats.items():
            last_activity = stats.get('last_activity', 0)
            if last_activity > 0:
                inactive_time = time.time() - last_activity
                if inactive_time > 120:  # Неактивна более 2 минут
                    alerts.append({
                        'type': 'warning',
                        'message': f'Печь {oven_id} неактивна {inactive_time / 60:.1f} мин',
                        'timestamp': datetime.now().isoformat()
                    })

        # Проверяем загрузку TPU
        tpu_stats = self.camera_manager.tpu_pool.get_stats()
        if tpu_stats['queue_size'] > tpu_stats['total_capacity'] * 0.8:
            alerts.append({
                'type': 'warning',
                'message': 'Высокая загрузка TPU пула',
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _get_system_uptime(self):
        """Время работы системы (заглушка)"""
        # В реальной системе здесь будет время запуска
        return {
            'uptime_seconds': 3600,  # 1 час для примера
            'uptime_formatted': '1 час 0 минут'
        }

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Запуск веб-сервера"""
        print(f"🌐 Запуск веб-сервера на http://{host}:{port}")
        print("📊 Dashboard доступен по адресу: http://localhost:5000")

        self.app.run(host=host, port=port, debug=debug, threaded=True)


# Отдельная функция для запуска API в потоке
def start_monitoring_api(camera_manager, host='0.0.0.0', port=5000):
    """Запуск API мониторинга в отдельном потоке"""
    import threading

    api = MonitoringAPI(camera_manager)

    def run_api():
        api.run(host=host, port=port, debug=False)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    return api, api_thread


if __name__ == "__main__":
    # Тестовый запуск API (без камер)
    class MockCameraManager:
        def __init__(self):
            self.cameras = {1: None, 2: None}
            self.stats = {
                1: {'current_fps': 15, 'frames_processed': 1500, 'detections_count': 450, 'last_activity': time.time()},
                2: {'current_fps': 12, 'frames_processed': 1200, 'detections_count': 380, 'last_activity': time.time()}
            }
            self.counters = {}

            # Mock TPU pool
            class MockTPUPool:
                def get_stats(self):
                    return {'active_devices': 1, 'queue_size': 5, 'total_capacity': 100}

            self.tpu_pool = MockTPUPool()

        def get_system_overview(self):
            return {
                'total_cameras': 2,
                'active_cameras': 2,
                'active_batches': 1,
                'total_frames_processed': 2700,
                'total_detections': 830
            }

        def get_oven_status(self, oven_id):
            if oven_id in self.stats:
                stats = self.stats[oven_id]
                return {
                    'oven_id': oven_id,
                    'fps': stats['current_fps'],
                    'frames_processed': stats['frames_processed'],
                    'detections_count': stats['detections_count'],
                    'last_activity': stats['last_activity'],
                    'tracked_objects': 5
                }
            return {'error': 'Oven not found'}


    # Запуск тестового API
    mock_manager = MockCameraManager()
    api = MonitoringAPI(mock_manager)

    print("🧪 Запуск тестового веб-сервера...")
    api.run(debug=True)