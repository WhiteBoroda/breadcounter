import cv2
import os
import time
from datetime import datetime
import json
import threading


class DataCollector:
    """Сбор данных с камер для обучения модели"""

    def __init__(self, output_dir='training_data'):
        self.output_dir = output_dir
        self.cameras = {}
        self.collecting = False

        # Создаем структуру папок
        self.setup_directories()

    def setup_directories(self):
        """Создание структуры папок для данных"""
        dirs = [
            f'{self.output_dir}/images',
            f'{self.output_dir}/annotations',
            f'{self.output_dir}/videos',
            f'{self.output_dir}/test_images'
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def add_camera(self, oven_id, camera_ip, login, password):
        """Добавление камеры для сбора данных"""
        rtsp_url = f"rtsp://{login}:{password}@{camera_ip}/stream1"
        cap = cv2.VideoCapture(rtsp_url)

        if cap.isOpened():
            self.cameras[oven_id] = {
                'capture': cap,
                'camera_ip': camera_ip,
                'last_save': 0
            }
            print(f"✅ Камера печи {oven_id} подключена")
            return True
        else:
            print(f"❌ Не удалось подключиться к камере печи {oven_id}")
            return False

    def collect_training_data(self, duration_minutes=30, save_interval=5):
        """Сбор обучающих данных"""
        print(f"🎬 Начинаем сбор данных на {duration_minutes} минут")
        print(f"📸 Сохранение каждые {save_interval} секунд")

        self.collecting = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        # Запускаем потоки для каждой камеры
        threads = []
        for oven_id in self.cameras.keys():
            thread = threading.Thread(
                target=self._collect_from_camera,
                args=(oven_id, save_interval, end_time),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        # Ждем завершения
        try:
            while time.time() < end_time and self.collecting:
                remaining = int((end_time - time.time()) / 60)
                print(f"⏱️  Осталось {remaining} минут...")
                time.sleep(60)
        except KeyboardInterrupt:
            print("🛑 Остановка сбора данных...")

        self.collecting = False

        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()

        print("✅ Сбор данных завершен")
        self._generate_collection_report()

    def _collect_from_camera(self, oven_id, save_interval, end_time):
        """Сбор данных с конкретной камеры"""
        camera_info = self.cameras[oven_id]
        cap = camera_info['capture']

        while self.collecting and time.time() < end_time:
            ret, frame = cap.read()
            if ret:
                current_time = time.time()

                # Сохраняем кадр через заданный интервал
                if current_time - camera_info['last_save'] >= save_interval:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"oven_{oven_id}_{timestamp}.jpg"
                    filepath = os.path.join(self.output_dir, 'images', filename)

                    cv2.imwrite(filepath, frame)
                    camera_info['last_save'] = current_time

                    print(f"📸 Сохранен кадр: {filename}")

            time.sleep(0.1)

    def collect_marker_samples(self):
        """Сбор образцов маркеров для обучения"""
        print("🎯 Режим сбора маркеров")
        print("Поместите маркеры перед камерами и нажмите:")
        print("  [c] - сохранить круг")
        print("  [s] - сохранить квадрат")
        print("  [t] - сохранить треугольник")
        print("  [q] - выход")

        marker_counts = {'circle': 0, 'square': 0, 'triangle': 0}

        while True:
            for oven_id, camera_info in self.cameras.items():
                ret, frame = camera_info['capture'].read()
                if ret:
                    # Показываем кадр
                    cv2.putText(frame, f"Oven {oven_id}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f'Oven {oven_id}', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):  # Круг
                self._save_marker_sample('circle', marker_counts)
            elif key == ord('s'):  # Квадрат
                self._save_marker_sample('square', marker_counts)
            elif key == ord('t'):  # Треугольник
                self._save_marker_sample('triangle', marker_counts)
            elif key == ord('q'):  # Выход
                break

        cv2.destroyAllWindows()
        print(f"🎯 Собрано маркеров: {marker_counts}")

    def _save_marker_sample(self, marker_type, counts):
        """Сохранение образца маркера"""
        for oven_id, camera_info in self.cameras.items():
            ret, frame = camera_info['capture'].read()
            if ret:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"marker_{marker_type}_{oven_id}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, 'images', filename)

                cv2.imwrite(filepath, frame)
                counts[marker_type] += 1

                print(f"✅ Сохранен маркер {marker_type}: {filename}")

    def _generate_collection_report(self):
        """Генерация отчета о собранных данных"""
        images_dir = os.path.join(self.output_dir, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        report = {
            'total_images': len(image_files),
            'collection_date': datetime.now().isoformat(),
            'cameras': list(self.cameras.keys())
        }

        # Группировка по печам
        oven_counts = {}
        for filename in image_files:
            if filename.startswith('oven_'):
                oven_id = filename.split('_')[1]
                oven_counts[oven_id] = oven_counts.get(oven_id, 0) + 1

        report['images_per_oven'] = oven_counts

        # Сохраняем отчет
        report_path = os.path.join(self.output_dir, 'collection_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Отчет сохранен: {report_path}")
        print(f"🖼️  Всего изображений: {len(image_files)}")