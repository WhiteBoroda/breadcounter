# zone_training_module.py - Интегрированный модуль зонной разметки
"""
Модуль зонной разметки для системы обучения.
Интегрируется с существующей архитектурой improved_interactive_training_web.py
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import base64


class ZoneManager:
    """Менеджер зон для обучения (интеграция с существующей системой)"""

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.zones = {
            'counting_zone': None,  # Основная зона подсчета
            'entry_zone': None,  # Зона входа (из печи)
            'exit_zone': None,  # Зона выхода (на стол)
            'exclude_zones': []  # Зоны исключения
        }

        # Информация о партии (интегрируется с существующими bread_types)
        self.batch_info = {
            'name': '',
            'weight': 0.0,
            'target_count': 0,
            'bread_type': 'white_bread'  # Используем существующие типы
        }

        # Параметры детекции для зонной разметки
        self.detection_params = {
            'min_area': 2000,
            'max_area': 25000,
            'hsv_lower': [10, 20, 20],
            'hsv_upper': [30, 255, 200]
        }

    def load_zones_for_video(self, video_filename):
        """Загрузка зон для конкретного видео"""
        zones_file = self._get_zones_filename(video_filename)
        if os.path.exists(zones_file):
            try:
                with open(zones_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.zones = data.get('zones', self.zones)
                    self.batch_info = data.get('batch_info', self.batch_info)
                    self.detection_params = data.get('detection_params', self.detection_params)
                return True
            except Exception as e:
                print(f"Ошибка загрузки зон: {e}")
        return False

    def save_zones_for_video(self, video_filename):
        """Сохранение зон для конкретного видео"""
        zones_file = self._get_zones_filename(video_filename)
        os.makedirs(os.path.dirname(zones_file), exist_ok=True)

        save_data = {
            'zones': self.zones,
            'batch_info': self.batch_info,
            'detection_params': self.detection_params,
            'video_filename': video_filename,
            'created': datetime.now().isoformat(),
            'version': '2.0'
        }

        try:
            with open(zones_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Ошибка сохранения зон: {e}")
            return False

    def detect_objects_in_zones(self, frame):
        """Детекция объектов с учетом зон"""
        detections = self._detect_bread_objects(frame)

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
            detection['bread_type'] = self.batch_info.get('bread_type', 'white_bread')

            filtered_detections.append(detection)

        return filtered_detections

    def generate_training_annotations(self, frame, frame_index, detections):
        """Генерация аннотаций в формате существующей системы"""
        annotations = []

        # Фильтруем только объекты в зоне подсчета
        valid_detections = [d for d in detections if d.get('in_counting_zone', False)]

        for detection in valid_detections:
            x1, y1, x2, y2 = detection['bbox']

            # Формат аннотации совместимый с existing системой
            annotation = {
                'bbox': [x1, y1, x2, y2],
                'center': detection['center'],
                'bread_type': detection['bread_type'],
                'confidence': detection.get('confidence', 0.8),
                'area': detection.get('area', 0),
                'zone_validated': True,  # Помечаем что прошло зонную валидацию
                'batch_info': self.batch_info
            }
            annotations.append(annotation)

        return annotations

    def create_zone_training_dataset(self, video_cap, frames_count=200):
        """Создание датасета с зонной разметкой (интеграция с existing pipeline)"""
        if not video_cap:
            return {'success': False, 'error': 'Video not loaded'}

        if not self.zones.get('counting_zone'):
            return {'success': False, 'error': 'Counting zone not defined'}

        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // frames_count)

        generated_annotations = []
        processed_frames = 0

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
            annotations = self.generate_training_annotations(frame, frame_idx, detections)

            if len(annotations) > 0:
                # Сохраняем в формате existing системы
                frame_data = {
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / video_cap.get(cv2.CAP_PROP_FPS),
                    'annotations': annotations,
                    'bread_type': self.batch_info['bread_type'],
                    'batch_info': self.batch_info,
                    'zone_metadata': {
                        'zones_used': list(self.zones.keys()),
                        'detection_params': self.detection_params
                    }
                }
                generated_annotations.append(frame_data)
                processed_frames += 1

        return {
            'success': True,
            'generated_frames': processed_frames,
            'total_annotations': sum(len(f['annotations']) for f in generated_annotations),
            'annotations_data': generated_annotations
        }

    def visualize_zones_on_frame(self, frame, detections=None):
        """Визуализация зон и детекций на кадре"""
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

        # Рисуем детекции если есть
        if detections:
            for detection in detections:
                self._draw_detection(annotated_frame, detection)

        # Добавляем информацию о партии
        if self.batch_info['name']:
            info_text = f"Партия: {self.batch_info['name']} ({self.batch_info['bread_type']})"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_frame

    def _detect_bread_objects(self, frame):
        """Базовая детекция объектов (упрощенная для зонной разметки)"""
        detections = []

        # HSV детекция
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.detection_params['hsv_lower'])
        upper = np.array(self.detection_params['hsv_upper'])
        mask = cv2.inRange(hsv, lower, upper)

        # Морфология
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < self.detection_params['min_area'] or area > self.detection_params['max_area']:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2

            detection = {
                'id': i,
                'bbox': [x, y, x + w, y + h],
                'center': [center_x, center_y],
                'area': area,
                'confidence': min(0.95, 0.5 + (area / self.detection_params['max_area']) * 0.4)
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
        """Проверка попадания в зоны исключения"""
        for zone in self.zones.get('exclude_zones', []):
            if zone and len(zone) >= 3:
                points = np.array(zone, np.int32)
                if cv2.pointPolygonTest(points, (x, y), False) >= 0:
                    return True
        return False

    def _draw_zone(self, frame, zone, color):
        """Отрисовка зоны"""
        if not zone or len(zone) < 3:
            return
        points = np.array(zone, np.int32)

        # Полупрозрачная заливка
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Контур
        cv2.polylines(frame, [points], True, color, 2)

    def _draw_detection(self, frame, detection):
        """Отрисовка детекции"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        center_x, center_y = detection['center']

        # Цвет в зависимости от зоны
        if detection.get('in_counting_zone'):
            color = (0, 255, 0)  # Зеленый
            thickness = 3
        elif detection.get('in_entry_zone'):
            color = (255, 0, 0)  # Синий
            thickness = 2
        elif detection.get('in_exit_zone'):
            color = (0, 0, 255)  # Красный
            thickness = 2
        else:
            color = (255, 255, 255)  # Белый
            thickness = 1

        # Рамка и центр
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)

        # Информация
        label = f"ID: {detection['id']} ({detection['confidence']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _get_zones_filename(self, video_filename):
        """Путь к файлу зон для видео"""
        video_name = Path(video_filename).stem
        return f"training_data/zones/{video_name}_zones.json"


# Интеграция с Flask routes для добавления к existing системе
def add_zone_routes_to_app(app, zone_manager):
    """Добавление маршрутов зонной разметки к existing Flask app"""

    @app.route('/api/zones/load', methods=['POST'])
    def load_zones():
        data = request.get_json()
        video_filename = data.get('video_filename')

        if not video_filename:
            return jsonify({'success': False, 'error': 'Video filename required'})

        success = zone_manager.load_zones_for_video(video_filename)
        return jsonify({
            'success': success,
            'zones': zone_manager.zones if success else {},
            'batch_info': zone_manager.batch_info if success else {}
        })

    @app.route('/api/zones/save', methods=['POST'])
    def save_zones():
        data = request.get_json()
        video_filename = data.get('video_filename')
        zones = data.get('zones', {})
        batch_info = data.get('batch_info', {})

        if not video_filename:
            return jsonify({'success': False, 'error': 'Video filename required'})

        zone_manager.zones = zones
        zone_manager.batch_info = batch_info

        success = zone_manager.save_zones_for_video(video_filename)
        return jsonify({'success': success})

    @app.route('/api/zones/generate_dataset', methods=['POST'])
    def generate_zone_dataset():
        data = request.get_json()
        frames_count = data.get('frames_count', 200)
        video_cap = getattr(app, 'current_video_cap', None)  # Получаем из existing системы

        if not video_cap:
            return jsonify({'success': False, 'error': 'No video loaded'})

        result = zone_manager.create_zone_training_dataset(video_cap, frames_count)
        return jsonify(result)

    @app.route('/api/zones/visualize_frame', methods=['POST'])
    def visualize_frame_with_zones():
        data = request.get_json()
        frame_index = data.get('frame_index', 0)
        video_cap = getattr(app, 'current_video_cap', None)

        if not video_cap:
            return jsonify({'success': False, 'error': 'No video loaded'})

        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_cap.read()

        if not ret:
            return jsonify({'success': False, 'error': 'Could not read frame'})

        # Детекция и визуализация
        detections = zone_manager.detect_objects_in_zones(frame)
        annotated_frame = zone_manager.visualize_zones_on_frame(frame, detections)

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
            }
        })


# JavaScript код для интеграции в existing веб-интерфейс
ZONE_INTERFACE_JS = '''
// zone_interface.js - Интеграция зонной разметки в existing интерфейс

class ZoneInterface {
    constructor() {
        this.zones = {
            counting_zone: null,
            entry_zone: null, 
            exit_zone: null,
            exclude_zones: []
        };
        this.currentTool = null;
        this.currentZone = [];
        this.isDrawing = false;
    }

    initializeZoneTools() {
        // Добавляем кнопки зонных инструментов к existing интерфейсу
        const toolsContainer = document.getElementById('annotation-tools') || 
                              document.querySelector('.control-panel');

        if (toolsContainer) {
            const zoneToolsHTML = `
                <div class="zone-tools" style="margin-top: 20px; padding: 15px; border: 2px solid #3498db; border-radius: 8px;">
                    <h4>🎯 Зонная разметка:</h4>
                    <button class="zone-btn" data-zone="counting_zone">🟢 Зона подсчета</button>
                    <button class="zone-btn" data-zone="entry_zone">🔵 Зона входа</button>
                    <button class="zone-btn" data-zone="exit_zone">🔴 Зона выхода</button>
                    <button class="zone-btn" data-zone="exclude_zone">⚫ Исключение</button>
                    <button class="zone-btn" id="loadZones">📂 Загрузить зоны</button>
                    <button class="zone-btn" id="saveZones">💾 Сохранить зоны</button>
                    <button class="zone-btn" id="generateZoneDataset">🚀 Создать датасет</button>
                </div>
            `;
            toolsContainer.insertAdjacentHTML('beforeend', zoneToolsHTML);

            this.bindZoneEvents();
        }
    }

    bindZoneEvents() {
        // Привязываем события к кнопкам зон
        document.querySelectorAll('.zone-btn[data-zone]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectZoneTool(e.target.dataset.zone);
            });
        });

        document.getElementById('loadZones')?.addEventListener('click', () => this.loadZones());
        document.getElementById('saveZones')?.addEventListener('click', () => this.saveZones());
        document.getElementById('generateZoneDataset')?.addEventListener('click', () => this.generateZoneDataset());

        // Интеграция с existing canvas events
        const canvas = document.getElementById('video-canvas') || document.querySelector('canvas');
        if (canvas) {
            canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
            canvas.addEventListener('dblclick', (e) => this.finishZone(e));
        }

        // ESC для отмены
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.cancelDrawing();
            }
        });
    }

    selectZoneTool(zoneType) {
        this.currentTool = zoneType;
        this.currentZone = [];
        this.isDrawing = false;

        // Обновляем UI
        document.querySelectorAll('.zone-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-zone="${zoneType}"]`)?.classList.add('active');

        this.showStatus(`Рисование зоны: ${this.getZoneLabel(zoneType)}`, 'info');
    }

    handleCanvasClick(event) {
        if (!this.currentTool) return;

        const canvas = event.target;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Масштабируем координаты к размеру видео
        const scaleX = canvas.videoWidth / rect.width;
        const scaleY = canvas.videoHeight / rect.height;

        const imgX = Math.round(x * scaleX);
        const imgY = Math.round(y * scaleY);

        this.currentZone.push([imgX, imgY]);
        this.isDrawing = true;

        this.redrawZones();
    }

    finishZone(event) {
        event.preventDefault();

        if (!this.isDrawing || this.currentZone.length < 3) return;

        if (this.currentTool === 'exclude_zone') {
            this.zones.exclude_zones.push([...this.currentZone]);
        } else {
            this.zones[this.currentTool] = [...this.currentZone];
        }

        this.currentZone = [];
        this.isDrawing = false;
        this.currentTool = null;

        document.querySelectorAll('.zone-btn').forEach(btn => btn.classList.remove('active'));

        this.redrawZones();
        this.showStatus('Зона сохранена', 'success');
    }

    async loadZones() {
        const currentVideo = getCurrentVideoFilename(); // Функция из existing системы
        if (!currentVideo) {
            this.showStatus('Сначала загрузите видео', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/zones/load', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video_filename: currentVideo})
            });

            const data = await response.json();
            if (data.success) {
                this.zones = data.zones;
                this.redrawZones();
                this.showStatus('Зоны загружены', 'success');
            } else {
                this.showStatus('Зоны не найдены', 'info');
            }
        } catch (error) {
            this.showStatus('Ошибка загрузки зон: ' + error.message, 'error');
        }
    }

    async saveZones() {
        const currentVideo = getCurrentVideoFilename();
        if (!currentVideo) {
            this.showStatus('Сначала загрузите видео', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/zones/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    video_filename: currentVideo,
                    zones: this.zones,
                    batch_info: this.getBatchInfo()
                })
            });

            const data = await response.json();
            if (data.success) {
                this.showStatus('Зоны сохранены', 'success');
            } else {
                this.showStatus('Ошибка сохранения', 'error');
            }
        } catch (error) {
            this.showStatus('Ошибка: ' + error.message, 'error');
        }
    }

    async generateZoneDataset() {
        if (!this.zones.counting_zone) {
            this.showStatus('Сначала создайте зону подсчета', 'warning');
            return;
        }

        const framesCount = prompt('Количество кадров для генерации:', '200');
        if (!framesCount) return;

        this.showStatus('Генерация датасета...', 'info');

        try {
            const response = await fetch('/api/zones/generate_dataset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frames_count: parseInt(framesCount)})
            });

            const data = await response.json();
            if (data.success) {
                this.showStatus(`Датасет создан: ${data.generated_frames} кадров, ${data.total_annotations} объектов`, 'success');
            } else {
                this.showStatus('Ошибка создания датасета: ' + data.error, 'error');
            }
        } catch (error) {
            this.showStatus('Ошибка: ' + error.message, 'error');
        }
    }

    redrawZones() {
        // Интеграция с existing canvas rendering
        const canvas = document.getElementById('video-canvas') || document.querySelector('canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Очищаем старые зоны (предполагаем что existing система перерисует основной контент)
        this.drawZonesOnCanvas(ctx, canvas);
    }

    drawZonesOnCanvas(ctx, canvas) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = rect.width / (canvas.videoWidth || canvas.width);
        const scaleY = rect.height / (canvas.videoHeight || canvas.height);

        const zoneColors = {
            counting_zone: 'rgba(0, 255, 0, 0.3)',
            entry_zone: 'rgba(255, 0, 0, 0.3)',
            exit_zone: 'rgba(0, 0, 255, 0.3)',
            exclude_zones: 'rgba(128, 128, 128, 0.3)'
        };

        // Рисуем зоны
        Object.entries(zoneColors).forEach(([zoneName, color]) => {
            if (zoneName === 'exclude_zones') {
                this.zones.exclude_zones.forEach(zone => {
                    this.drawZonePolygon(ctx, zone, color, scaleX, scaleY);
                });
            } else {
                const zone = this.zones[zoneName];
                if (zone) {
                    this.drawZonePolygon(ctx, zone, color, scaleX, scaleY);
                }
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
            exit_zone: 'Зона выхода',
            exclude_zone: 'Зона исключения'
        };
        return labels[zoneType] || zoneType;
    }

    getBatchInfo() {
        // Интеграция с existing формами ввода информации о партии
        return {
            name: document.getElementById('batch-name')?.value || '',
            weight: parseFloat(document.getElementById('batch-weight')?.value) || 0,
            target_count: parseInt(document.getElementById('target-count')?.value) || 0,
            bread_type: document.getElementById('bread-type')?.value || 'white_bread'
        };
    }

    showStatus(message, type = 'info') {
        // Интеграция с existing системой показа статусов
        if (window.showStatus) {
            window.showStatus(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.zoneInterface === 'undefined') {
        window.zoneInterface = new ZoneInterface();
        window.zoneInterface.initializeZoneTools();
    }
});
'''

if __name__ == '__main__':
    # Пример использования для интеграции с existing системой
    zone_manager = ZoneManager()

    print("🎯 Модуль зонной разметки инициализирован")
    print("📋 Для интеграции с existing Flask app:")
    print("   from zone_training_module import add_zone_routes_to_app, ZoneManager")
    print("   zone_manager = ZoneManager()")
    print("   add_zone_routes_to_app(app, zone_manager)")
    print("📱 Добавьте ZONE_INTERFACE_JS в веб-интерфейс")