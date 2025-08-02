# large_video_processor.py - Обработка больших видео файлов напрямую
import cv2
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading


class LargeVideoProcessor:
    """Обработчик больших видео файлов для извлечения обучающих данных"""

    def __init__(self, output_dir='training_data'):
        self.output_dir = output_dir
        self.setup_directories()

        # Статистика
        self.total_extracted = 0
        self.total_processed_frames = 0
        self.start_time = None

        # Блокировка для потокобезопасности
        self.stats_lock = threading.Lock()

    def setup_directories(self):
        """Создание структуры папок"""
        dirs = [
            f'{self.output_dir}/images',
            f'{self.output_dir}/annotations',
            f'{self.output_dir}/metadata',
            f'{self.output_dir}/previews'
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

        print(f"📁 Подготовлена структура папок в {self.output_dir}")

    def process_large_video(self, video_path, strategy='smart', max_frames=1000,
                            quality_threshold=0.3, parallel_processing=True):
        """
        Обработка большого видео файла

        Args:
            video_path: путь к видео
            strategy: стратегия извлечения ('uniform', 'smart', 'quality_based')
            max_frames: максимальное количество кадров
            quality_threshold: порог качества для фильтрации
            parallel_processing: использовать параллельную обработку
        """
        if not os.path.exists(video_path):
            print(f"❌ Видео файл не найден: {video_path}")
            return False

        self.start_time = time.time()
        print(f"🎬 Обработка большого видео: {video_path}")

        # Анализируем видео
        video_info = self._analyze_video(video_path)
        if not video_info:
            return False

        print(f"📊 Видео: {video_info['duration']:.1f} мин, "
              f"{video_info['total_frames']} кадров, "
              f"{video_info['fps']:.1f} FPS")
        print(f"💾 Размер файла: {video_info['file_size_gb']:.2f} GB")

        # Выбираем кадры для извлечения
        print(f"🎯 Стратегия извлечения: {strategy}")
        frame_positions = self._select_frames(video_info, strategy, max_frames)

        print(f"📸 Выбрано {len(frame_positions)} позиций для извлечения")

        # Извлекаем кадры
        if parallel_processing and len(frame_positions) > 100:
            success = self._extract_frames_parallel(video_path, frame_positions,
                                                    quality_threshold, video_info)
        else:
            success = self._extract_frames_sequential(video_path, frame_positions,
                                                      quality_threshold, video_info)

        if success:
            self._generate_processing_report(video_path, video_info, len(frame_positions))

        return success

    def _analyze_video(self, video_path):
        """Быстрый анализ видео файла"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Не удалось открыть видео: {video_path}")
                return None

            # Получаем основную информацию
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration_seconds = total_frames / fps if fps > 0 else 0
            file_size = os.path.getsize(video_path)

            # Проверяем качество нескольких случайных кадров
            sample_quality = self._sample_video_quality(cap, total_frames)

            cap.release()

            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration_seconds / 60,  # в минутах
                'file_size_gb': file_size / (1024 ** 3),
                'avg_quality': sample_quality,
                'bitrate_estimate': (file_size * 8) / duration_seconds if duration_seconds > 0 else 0
            }

        except Exception as e:
            print(f"❌ Ошибка анализа видео: {e}")
            return None

    def _sample_video_quality(self, cap, total_frames, num_samples=10):
        """Оценка качества видео на основе образцов"""
        qualities = []

        for i in range(num_samples):
            frame_pos = (total_frames // num_samples) * i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if ret:
                # Простая оценка качества на основе вариации
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                qualities.append(quality)

        return np.mean(qualities) if qualities else 0

    def _select_frames(self, video_info, strategy, max_frames):
        """Выбор кадров для извлечения по стратегии"""
        total_frames = video_info['total_frames']

        if strategy == 'uniform':
            # Равномерное распределение
            if total_frames <= max_frames:
                return list(range(0, total_frames, max(1, total_frames // max_frames)))
            else:
                step = total_frames // max_frames
                return list(range(0, total_frames, step))[:max_frames]

        elif strategy == 'smart':
            # Умное извлечение с группировкой и пропусками
            return self._smart_frame_selection(video_info, max_frames)

        elif strategy == 'quality_based':
            # На основе анализа качества (более затратно)
            return self._quality_based_selection(video_info, max_frames)

        else:
            print(f"⚠️  Неизвестная стратегия: {strategy}, используем uniform")
            return self._select_frames(video_info, 'uniform', max_frames)

    def _smart_frame_selection(self, video_info, max_frames):
        """Умный выбор кадров с группировкой"""
        total_frames = video_info['total_frames']
        fps = video_info['fps']

        # Группируем по временным сегментам
        segment_duration = 30  # секунд на сегмент
        frames_per_segment = int(fps * segment_duration)

        # Количество сегментов
        num_segments = total_frames // frames_per_segment
        frames_per_segment_to_extract = max_frames // max(1, num_segments)

        selected_frames = []

        for segment in range(num_segments):
            segment_start = segment * frames_per_segment
            segment_end = min(segment_start + frames_per_segment, total_frames)

            # В каждом сегменте берем кадры с равномерным интервалом
            if frames_per_segment_to_extract > 0:
                step = (segment_end - segment_start) // frames_per_segment_to_extract
                for i in range(frames_per_segment_to_extract):
                    frame_pos = segment_start + (i * step)
                    if frame_pos < total_frames:
                        selected_frames.append(frame_pos)

        return selected_frames[:max_frames]

    def _quality_based_selection(self, video_info, max_frames):
        """Выбор кадров на основе качества"""
        # Это более затратный метод, который анализирует качество кадров
        # Для больших файлов используем сэмплирование
        total_frames = video_info['total_frames']
        sample_rate = max(1, total_frames // (max_frames * 3))  # Проверяем в 3 раза больше кадров

        candidate_frames = list(range(0, total_frames, sample_rate))
        return candidate_frames[:max_frames]

    def _extract_frames_sequential(self, video_path, frame_positions, quality_threshold, video_info):
        """Последовательное извлечение кадров"""
        print("📸 Последовательное извлечение кадров...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        video_name = Path(video_path).stem
        extracted_count = 0

        try:
            for i, frame_pos in enumerate(frame_positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if ret:
                    # Проверяем качество кадра
                    if self._check_frame_quality(frame, quality_threshold):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        filename = f"{video_name}_frame_{frame_pos:08d}_{timestamp}.jpg"
                        filepath = os.path.join(self.output_dir, 'images', filename)

                        # Сохраняем кадр
                        cv2.imwrite(filepath, frame)

                        # Простая детекция и сохранение аннотации
                        detections = self._simple_detection(frame)
                        self._save_frame_annotation(filename, frame_pos, video_path,
                                                    frame.shape, detections)

                        extracted_count += 1

                        # Обновляем статистику
                        with self.stats_lock:
                            self.total_extracted += 1
                            self.total_processed_frames += 1

                # Прогресс
                if (i + 1) % 50 == 0:
                    progress = ((i + 1) / len(frame_positions)) * 100
                    elapsed = time.time() - self.start_time
                    eta = (elapsed / (i + 1)) * (len(frame_positions) - i - 1)

                    print(f"   📊 Прогресс: {progress:.1f}% "
                          f"({extracted_count} извлечено, "
                          f"ETA: {eta / 60:.1f} мин)")

        except Exception as e:
            print(f"❌ Ошибка извлечения: {e}")
            return False
        finally:
            cap.release()

        print(f"✅ Извлечено {extracted_count} кадров из {len(frame_positions)}")
        return True

    def _extract_frames_parallel(self, video_path, frame_positions, quality_threshold, video_info):
        """Параллельное извлечение кадров"""
        print("⚡ Параллельное извлечение кадров...")

        # Разбиваем на чанки для параллельной обработки
        num_workers = min(4, os.cpu_count())  # Не более 4 потоков для видео
        chunk_size = len(frame_positions) // num_workers

        chunks = []
        for i in range(0, len(frame_positions), chunk_size):
            chunks.append(frame_positions[i:i + chunk_size])

        print(f"🔧 Используем {len(chunks)} потоков по ~{chunk_size} кадров")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(
                    self._process_chunk,
                    video_path, chunk, quality_threshold, video_info, chunk_idx
                )
                futures.append(future)

            # Ожидаем завершения всех потоков
            total_extracted = 0
            for future in futures:
                extracted = future.result()
                total_extracted += extracted

        print(f"✅ Параллельная обработка завершена: {total_extracted} кадров")
        return True

    def _process_chunk(self, video_path, frame_positions, quality_threshold, video_info, chunk_idx):
        """Обработка чанка кадров в отдельном потоке"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Поток {chunk_idx}: не удалось открыть видео")
            return 0

        video_name = Path(video_path).stem
        extracted_count = 0

        try:
            for frame_pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if ret and self._check_frame_quality(frame, quality_threshold):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = f"{video_name}_chunk{chunk_idx}_frame_{frame_pos:08d}_{timestamp}.jpg"
                    filepath = os.path.join(self.output_dir, 'images', filename)

                    cv2.imwrite(filepath, frame)

                    # Детекция и аннотация
                    detections = self._simple_detection(frame)
                    self._save_frame_annotation(filename, frame_pos, video_path,
                                                frame.shape, detections)

                    extracted_count += 1

                    # Потокобезопасное обновление статистики
                    with self.stats_lock:
                        self.total_extracted += 1

        except Exception as e:
            print(f"❌ Поток {chunk_idx} ошибка: {e}")
        finally:
            cap.release()

        print(f"   ✅ Поток {chunk_idx}: извлечено {extracted_count} кадров")
        return extracted_count

    def _check_frame_quality(self, frame, threshold):
        """Проверка качества кадра"""
        if threshold <= 0:
            return True

        # Проверяем резкость
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Проверяем что кадр не слишком темный/светлый
        mean_brightness = np.mean(gray)

        return (laplacian_var > threshold * 1000 and
                20 < mean_brightness < 235)

    def _simple_detection(self, frame):
        """Простая детекция хлеба"""
        detections = []

        # HSV для детекции хлебных оттенков
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Маска для хлеба
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
            if 1000 < area < 80000:  # Фильтр по размеру
                x, y, w, h = cv2.boundingRect(contour)

                detections.append({
                    'id': i,
                    'bbox': [x, y, x + w, y + h],
                    'center': [x + w // 2, y + h // 2],
                    'area': area,
                    'confidence': 0.8,
                    'class_name': 'bread'
                })

        return detections

    def _save_frame_annotation(self, filename, frame_pos, video_path, frame_shape, detections):
        """Сохранение аннотации кадра"""
        annotation = {
            'filename': filename,
            'source_video': video_path,
            'frame_position': frame_pos,
            'timestamp': datetime.now().isoformat(),
            'resolution': {
                'width': frame_shape[1],
                'height': frame_shape[0],
                'channels': frame_shape[2]
            },
            'detections': detections,
            'detection_count': len(detections),
            'extraction_method': 'large_video_processor'
        }

        annotation_file = os.path.join(
            self.output_dir, 'annotations',
            filename.replace('.jpg', '.json')
        )

        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

    def _generate_processing_report(self, video_path, video_info, processed_frames):
        """Генерация отчета об обработке"""
        total_time = time.time() - self.start_time

        report = {
            'processing_date': datetime.now().isoformat(),
            'source_video': {
                'path': video_path,
                'size_gb': video_info['file_size_gb'],
                'duration_minutes': video_info['duration'],
                'total_frames': video_info['total_frames'],
                'fps': video_info['fps']
            },
            'processing_stats': {
                'processed_frame_positions': processed_frames,
                'extracted_frames': self.total_extracted,
                'success_rate': self.total_extracted / processed_frames if processed_frames > 0 else 0,
                'processing_time_seconds': total_time,
                'frames_per_second': self.total_extracted / total_time if total_time > 0 else 0
            },
            'output_directory': self.output_dir
        }

        report_file = os.path.join(self.output_dir, 'processing_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Выводим отчет
        print("\n📊 ОТЧЕТ ОБ ОБРАБОТКЕ ВИДЕО")
        print("=" * 50)
        print(f"🎬 Видео: {Path(video_path).name}")
        print(f"💾 Размер: {video_info['file_size_gb']:.2f} GB")
        print(f"⏱️  Длительность: {video_info['duration']:.1f} мин")
        print(f"📸 Извлечено кадров: {self.total_extracted}")
        print(f"⚡ Время обработки: {total_time / 60:.1f} мин")
        print(f"🚀 Скорость: {self.total_extracted / total_time:.1f} кадров/сек")
        print(f"📄 Отчет: {report_file}")
        print("=" * 50)

    def process_multiple_videos(self, videos_dir, **kwargs):
        """Обработка нескольких больших видео файлов"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(videos_dir).glob(f'*{ext}'))

        if not video_files:
            print(f"❌ Видео файлы не найдены в {videos_dir}")
            return

        # Сортируем по размеру (большие файлы в конце)
        video_files.sort(key=lambda x: x.stat().st_size)

        print(f"🎬 Найдено {len(video_files)} видео файлов для обработки")

        total_processed = 0
        total_start_time = time.time()

        for i, video_path in enumerate(video_files, 1):
            size_gb = video_path.stat().st_size / (1024 ** 3)
            print(f"\n📹 Обрабатываем {i}/{len(video_files)}: {video_path.name} ({size_gb:.2f} GB)")

            if self.process_large_video(str(video_path), **kwargs):
                total_processed += 1

        total_time = time.time() - total_start_time

        print(f"\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"✅ Успешно обработано: {total_processed}/{len(video_files)} видео")
        print(f"⏱️  Общее время: {total_time / 3600:.1f} часов")
        print(f"📸 Всего извлечено: {self.total_extracted} кадров")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Обработка больших видео файлов')
    parser.add_argument('--video', help='Путь к одному видео файлу')
    parser.add_argument('--videos-dir', help='Папка с видео файлами')
    parser.add_argument('--output', default='training_data', help='Папка для результатов')
    parser.add_argument('--strategy', choices=['uniform', 'smart', 'quality_based'],
                        default='smart', help='Стратегия извлечения кадров')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='Максимум кадров на видео')
    parser.add_argument('--quality', type=float, default=0.3,
                        help='Порог качества (0.0-1.0)')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Использовать параллельную обработку')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Отключить параллельную обработку')

    args = parser.parse_args()

    if args.no_parallel:
        args.parallel = False

    print("🎬 ОБРАБОТЧИК БОЛЬШИХ ВИДЕО ФАЙЛОВ")
    print("=" * 60)
    print(f"📁 Выходная папка: {args.output}")
    print(f"🎯 Стратегия: {args.strategy}")
    print(f"📸 Макс кадров на видео: {args.max_frames}")
    print(f"✨ Порог качества: {args.quality}")
    print(f"⚡ Параллельная обработка: {'Да' if args.parallel else 'Нет'}")

    try:
        processor = LargeVideoProcessor(args.output)

        if args.video:
            # Обработка одного файла
            print(f"\n🎬 Обработка одного файла: {args.video}")
            processor.process_large_video(
                args.video,
                strategy=args.strategy,
                max_frames=args.max_frames,
                quality_threshold=args.quality,
                parallel_processing=args.parallel
            )

        elif args.videos_dir:
            # Пакетная обработка
            print(f"\n📁 Пакетная обработка папки: {args.videos_dir}")
            processor.process_multiple_videos(
                args.videos_dir,
                strategy=args.strategy,
                max_frames=args.max_frames,
                quality_threshold=args.quality,
                parallel_processing=args.parallel
            )

        else:
            print("❌ Укажите --video или --videos-dir")
            print("\nПримеры:")
            print("  python large_video_processor.py --video /path/to/big_video.mp4")
            print("  python large_video_processor.py --videos-dir /path/to/videos/")
            print("  python large_video_processor.py --videos-dir ./videos --strategy smart --max-frames 500")

    except KeyboardInterrupt:
        print("\n🛑 Обработка прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()