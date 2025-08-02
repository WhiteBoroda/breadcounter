# training_pipeline.py - Полный пайплайн обучения модели
import cv2
import os
import time
import json
import threading
from datetime import datetime
from config_loader import ConfigLoader


class DataCollector:
    """Сбор данных с камер для обучения модели"""

    def __init__(self, config_file='cameras.yaml', output_dir='training_data'):
        self.config = ConfigLoader(config_file)
        self.cameras = {cam.oven_id: cam for cam in self.config.get_cameras()}
        self.output_dir = output_dir
        self.connections = {}
        self.collecting = False

        # Создаем структуру папок
        self.setup_directories()

    def setup_directories(self):
        """Создание структуры папок для данных"""
        dirs = [
            f'{self.output_dir}/images',
            f'{self.output_dir}/annotations',
            f'{self.output_dir}/videos',
            f'{self.output_dir}/test_images',
            f'{self.output_dir}/markers'
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

        print(f"📁 Создана структура папок в {self.output_dir}")

    def connect_cameras(self):
        """Подключение ко всем камерам из конфигурации"""
        print("🔌 Подключение к камерам...")

        connected_count = 0

        for oven_id, camera_config in self.cameras.items():
            rtsp_paths = [
                f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/stream1",
                f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/stream0",
                f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/live"
            ]

            for rtsp_url in rtsp_paths:
                try:
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 15)

                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            self.connections[oven_id] = {
                                'capture': cap,
                                'config': camera_config,
                                'last_save': 0,
                                'frame_count': 0
                            }
                            print(f"✅ {camera_config.oven_name}: подключена")
                            connected_count += 1
                            break
                    else:
                        cap.release()

                except Exception as e:
                    print(f"❌ Ошибка подключения к {camera_config.oven_name}: {e}")

        print(f"📊 Подключено камер: {connected_count}/{len(self.cameras)}")
        return connected_count > 0

    def collect_training_data(self, duration_minutes=20, save_interval=5):
        """Автоматический сбор обучающих данных"""
        if not self.connections:
            print("❌ Нет подключенных камер")
            return

        print(f"🎬 Начинаем сбор данных на {duration_minutes} минут")
        print(f"📸 Сохранение каждые {save_interval} секунд")
        print("   Убедитесь что печи активно работают!")

        input("Нажмите Enter для начала сбора данных...")

        self.collecting = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        # Запускаем потоки для каждой камеры
        threads = []
        for oven_id in self.connections.keys():
            thread = threading.Thread(
                target=self._collect_from_camera,
                args=(oven_id, save_interval, end_time),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        # Мониторинг прогресса
        try:
            while time.time() < end_time and self.collecting:
                remaining = int((end_time - time.time()) / 60)
                print(f"⏱️  Осталось {remaining} минут... ({self._get_collection_stats()})")
                time.sleep(30)  # Обновление каждые 30 секунд

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
        connection = self.connections[oven_id]
        cap = connection['capture']
        config = connection['config']

        while self.collecting and time.time() < end_time:
            ret, frame = cap.read()
            if ret:
                current_time = time.time()

                # Сохраняем кадр через заданный интервал
                if current_time - connection['last_save'] >= save_interval:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = f"oven_{oven_id}_{timestamp}.jpg"
                    filepath = os.path.join(self.output_dir, 'images', filename)

                    # Сохраняем с метаданными
                    cv2.imwrite(filepath, frame)

                    # Сохраняем метаданные
                    self._save_frame_metadata(filename, oven_id, config, frame.shape)

                    connection['last_save'] = current_time
                    connection['frame_count'] += 1

                    print(f"📸 {config.oven_name}: сохранен кадр {connection['frame_count']}")

            time.sleep(0.1)

    def _save_frame_metadata(self, filename, oven_id, config, frame_shape):
        """Сохранение метаданных кадра"""
        metadata = {
            'filename': filename,
            'oven_id': oven_id,
            'oven_name': config.oven_name,
            'camera_ip': config.camera_ip,
            'timestamp': datetime.now().isoformat(),
            'resolution': {
                'width': frame_shape[1],
                'height': frame_shape[0],
                'channels': frame_shape[2]
            },
            'workshop': config.workshop_name,
            'enterprise': config.enterprise_name
        }

        metadata_file = os.path.join(self.output_dir, 'images',
                                     filename.replace('.jpg', '_metadata.json'))

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def collect_marker_samples(self):
        """Интерактивный сбор образцов маркеров"""
        if not self.connections:
            print("❌ Нет подключенных камер")
            return

        print("🎯 РЕЖИМ СБОРА МАРКЕРОВ")
        print("=" * 50)
        print("Разместите маркеры перед камерами и нажимайте:")
        print("  [c] - сохранить КРУГ")
        print("  [s] - сохранить КВАДРАТ")
        print("  [t] - сохранить ТРЕУГОЛЬНИК")
        print("  [d] - сохранить РОМБ")
        print("  [w] - сохранить ЗВЕЗДУ")
        print("  [q] - выход из режима маркеров")
        print("=" * 50)

        marker_counts = {'circle': 0, 'square': 0, 'triangle': 0, 'diamond': 0, 'star': 0}
        target_per_marker = 20  # Целевое количество образцов каждого маркера

        while True:
            # Показываем кадры со всех камер
            display_frames = []

            for oven_id, connection in self.connections.items():
                ret, frame = connection['capture'].read()
                if ret:
                    config = connection['config']

                    # Добавляем информацию на кадр
                    info_frame = frame.copy()
                    cv2.putText(info_frame, f"{config.oven_name}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Показываем статистику маркеров
                    y_offset = 60
                    for marker_type, count in marker_counts.items():
                        color = (0, 255, 0) if count >= target_per_marker else (0, 255, 255)
                        text = f"{marker_type}: {count}/{target_per_marker}"
                        cv2.putText(info_frame, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 25

                    display_frames.append((oven_id, info_frame))

            # Показываем все кадры
            for oven_id, frame in display_frames:
                cv2.imshow(f'Markers - Oven {oven_id}', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):  # Круг
                self._save_marker_sample('circle', marker_counts)
            elif key == ord('s'):  # Квадрат
                self._save_marker_sample('square', marker_counts)
            elif key == ord('t'):  # Треугольник
                self._save_marker_sample('triangle', marker_counts)
            elif key == ord('d'):  # Ромб
                self._save_marker_sample('diamond', marker_counts)
            elif key == ord('w'):  # Звезда
                self._save_marker_sample('star', marker_counts)
            elif key == ord('q'):  # Выход
                break

        cv2.destroyAllWindows()

        print(f"\n🎯 Собрано маркеров:")
        for marker_type, count in marker_counts.items():
            status = "✅" if count >= target_per_marker else "⚠️"
            print(f"   {status} {marker_type}: {count}")

    def _save_marker_sample(self, marker_type, counts):
        """Сохранение образца маркера"""
        saved_count = 0

        for oven_id, connection in self.connections.items():
            ret, frame = connection['capture'].read()
            if ret:
                config = connection['config']
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f"marker_{marker_type}_oven{oven_id}_{timestamp}.jpg"

                # Сохраняем в отдельную папку для маркеров
                filepath = os.path.join(self.output_dir, 'markers', filename)
                cv2.imwrite(filepath, frame)

                # Также копируем в общую папку изображений
                main_filepath = os.path.join(self.output_dir, 'images', filename)
                cv2.imwrite(main_filepath, frame)

                # Сохраняем метаданные
                self._save_frame_metadata(filename, oven_id, config, frame.shape)

                saved_count += 1

        counts[marker_type] += saved_count
        print(f"✅ Сохранено образцов {marker_type}: {saved_count}")

    def _get_collection_stats(self):
        """Получение статистики сбора"""
        total_frames = sum(conn['frame_count'] for conn in self.connections.values())
        return f"кадров собрано: {total_frames}"

    def _generate_collection_report(self):
        """Генерация отчета о собранных данных"""
        images_dir = os.path.join(self.output_dir, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        # Статистика по печам
        oven_stats = {}
        marker_stats = {}

        for filename in image_files:
            if filename.startswith('oven_'):
                oven_id = filename.split('_')[1]
                oven_stats[oven_id] = oven_stats.get(oven_id, 0) + 1
            elif filename.startswith('marker_'):
                marker_type = filename.split('_')[1]
                marker_stats[marker_type] = marker_stats.get(marker_type, 0) + 1

        report = {
            'collection_date': datetime.now().isoformat(),
            'total_images': len(image_files),
            'images_per_oven': oven_stats,
            'marker_samples': marker_stats,
            'cameras_used': len(self.connections),
            'output_directory': self.output_dir
        }

        # Сохраняем отчет
        report_path = os.path.join(self.output_dir, 'collection_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Выводим отчет
        print("\n📊 ОТЧЕТ О СБОРЕ ДАННЫХ")
        print("=" * 40)
        print(f"🖼️  Всего изображений: {len(image_files)}")
        print(f"📹 Использовано камер: {len(self.connections)}")

        if oven_stats:
            print("\n📈 По печам:")
            for oven_id, count in oven_stats.items():
                print(f"   Печь {oven_id}: {count} кадров")

        if marker_stats:
            print("\n🎯 Маркеры:")
            for marker_type, count in marker_stats.items():
                print(f"   {marker_type}: {count} образцов")

        print(f"\n📄 Отчет сохранен: {report_path}")
        print("=" * 40)

    def cleanup(self):
        """Очистка ресурсов"""
        for connection in self.connections.values():
            if 'capture' in connection:
                connection['capture'].release()
        cv2.destroyAllWindows()


def run_training_pipeline():
    """Запуск полного пайплайна обучения"""
    print("🚀 ПАЙПЛАЙН ОБУЧЕНИЯ МОДЕЛИ ДЕТЕКЦИИ ХЛЕБА")
    print("=" * 60)

    try:
        # Инициализация сборщика данных
        collector = DataCollector()

        if not collector.connect_cameras():
            print("❌ Не удалось подключиться к камерам")
            print("   Проверьте настройки в cameras.yaml")
            return

        print("\n1️⃣ СБОР ОБУЧАЮЩИХ ДАННЫХ")
        print("-" * 30)

        # Выбор режима сбора данных
        print("Выберите режим сбора:")
        print("  [1] Автоматический сбор кадров")
        print("  [2] Сбор образцов маркеров")
        print("  [3] И то, и другое")
        print("  [0] Пропустить сбор данных")

        choice = input("Ваш выбор (1-3): ").strip()

        if choice in ['1', '3']:
            # Автоматический сбор
            print("\n🎬 Автоматический сбор данных")
            duration = input("Длительность сбора в минутах (по умолчанию 15): ").strip()
            duration = int(duration) if duration.isdigit() else 15

            interval = input("Интервал сохранения в секундах (по умолчанию 3): ").strip()
            interval = int(interval) if interval.isdigit() else 3

            collector.collect_training_data(duration_minutes=duration, save_interval=interval)

        if choice in ['2', '3']:
            # Сбор маркеров
            print("\n🎯 Сбор образцов маркеров")
            print("   Подготовьте маркеры: круг, квадрат, треугольник, ромб, звезда")

            ready = input("Маркеры готовы? (y/n): ").strip().lower()
            if ready == 'y':
                collector.collect_marker_samples()

        print("\n2️⃣ АННОТАЦИЯ ДАННЫХ")
        print("-" * 30)

        if choice != '0':
            annotate = input("Запустить инструмент аннотации? (y/n): ").strip().lower()
            if annotate == 'y':
                print("🏷️ Запуск инструмента аннотации...")
                print("   Откроется отдельное окно для разметки данных")

                from annotation_tool import AnnotationTool
                annotator = AnnotationTool('training_data/images', 'training_data/annotations')
                annotator.annotate_dataset()

        print("\n3️⃣ ОБУЧЕНИЕ МОДЕЛИ")
        print("-" * 30)

        train = input("Начать обучение модели? (y/n): ").strip().lower()
        if train == 'y':
            print("🧠 Обучение модели...")
            print("   Это может занять 30-60 минут")

            try:
                from model_trainer import BreadDetectionTrainer
                trainer = BreadDetectionTrainer('training_data')
                model, history = trainer.train_model(epochs=50)

                if model:
                    print("\n4️⃣ КОНВЕРТАЦИЯ ДЛЯ CORAL TPU")
                    print("-" * 30)

                    # Конвертация в TF Lite
                    tflite_path = trainer.convert_to_tflite('bread_detector_final.h5')

                    if tflite_path:
                        # Компиляция для Edge TPU
                        edge_tpu_path = trainer.compile_for_edge_tpu(tflite_path)

                        if edge_tpu_path:
                            print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
                            print("=" * 50)
                            print(f"🧠 Модель для Coral TPU: {edge_tpu_path}")
                            print(f"📁 Данные сохранены в: training_data/")
                            print("📄 Копируйте .tflite файл на устройство с Coral TPU")
                            print("\n🚀 Теперь можно запустить полную систему:")
                            print("   python main_multicamera.py cameras.yaml")

            except ImportError:
                print("⚠️  Модуль обучения не найден")
                print("   Установите: pip install tensorflow ultralytics")
            except Exception as e:
                print(f"❌ Ошибка обучения: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Пайплайн прерван пользователем")
    except Exception as e:
        print(f"❌ Ошибка в пайплайне: {e}")
    finally:
        if 'collector' in locals():
            collector.cleanup()

    print("\n✅ Пайплайн обучения завершен")


if __name__ == "__main__":
    run_training_pipeline()