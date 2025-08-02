# video_data_extractor.py - Извлечение кадров из MP4 записей
import cv2
import os
import time
import json
from datetime import datetime
from pathlib import Path


class VideoDataExtractor:
    """Извлечение обучающих кадров из MP4 записей"""

    def __init__(self, output_dir='training_data'):
        self.output_dir = output_dir
        self.setup_directories()

    def setup_directories(self):
        """Создание структуры папок для данных"""
        dirs = [
            f'{self.output_dir}/images',
            f'{self.output_dir}/annotations',
            f'{self.output_dir}/metadata'
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

        print(f"📁 Создана структура папок в {self.output_dir}")

    def extract_frames_from_video(self, video_path, oven_id=1, frame_interval=30, max_frames=1000):
        """
        Извлечение кадров из видеофайла

        Args:
            video_path: путь к MP4 файлу
            oven_id: ID печи для именования
            frame_interval: интервал между кадрами (каждый N-й кадр)
            max_frames: максимальное количество кадров
        """
        if not os.path.exists(video_path):
            print(f"❌ Видеофайл не найден: {video_path}")
            return False

        print(f"🎬 Извлекаем кадры из {video_path}")
        print(f"   📋 Параметры: каждый {frame_interval}-й кадр, макс {max_frames} кадров")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {video_path}")
            return False

        # Информация о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"   📊 Видео: {total_frames} кадров, {fps:.1f} FPS, {duration / 60:.1f} мин")

        frame_count = 0
        extracted_count = 0
        video_name = Path(video_path).stem

        try:
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Извлекаем каждый N-й кадр
                if frame_count % frame_interval == 0:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = f"video_{video_name}_oven{oven_id}_{extracted_count:04d}_{timestamp}.jpg"
                    filepath = os.path.join(self.output_dir, 'images', filename)

                    # Сохраняем кадр
                    cv2.imwrite(filepath, frame)

                    # Сохраняем метаданные
                    self._save_frame_metadata(
                        filename, oven_id, video_path,
                        frame_count, total_frames, fps, frame.shape
                    )

                    extracted_count += 1

                    if extracted_count % 50 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"   📸 Извлечено {extracted_count} кадров ({progress:.1f}%)")

                frame_count += 1

        except KeyboardInterrupt:
            print("🛑 Извлечение прервано пользователем")
        finally:
            cap.release()

        print(f"✅ Извлечено {extracted_count} кадров из {video_path}")
        return True

    def batch_extract_from_directory(self, videos_dir, frame_interval=30, max_frames_per_video=500):
        """
        Пакетное извлечение из всех MP4 файлов в папке

        Args:
            videos_dir: папка с MP4 файлами
            frame_interval: интервал между кадрами
            max_frames_per_video: максимум кадров с одного видео
        """
        if not os.path.exists(videos_dir):
            print(f"❌ Папка не найдена: {videos_dir}")
            return

        # Ищем все MP4 файлы
        video_files = []
        for ext in ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']:
            video_files.extend(Path(videos_dir).glob(ext))

        if not video_files:
            print(f"❌ MP4 файлы не найдены в {videos_dir}")
            return

        print(f"🎬 Найдено {len(video_files)} видеофайлов")

        total_extracted = 0
        for i, video_path in enumerate(video_files, 1):
            print(f"\n📹 Обрабатываем файл {i}/{len(video_files)}: {video_path.name}")

            # Пытаемся извлечь oven_id из имени файла
            oven_id = self._extract_oven_id_from_filename(video_path.name)

            if self.extract_frames_from_video(str(video_path), oven_id, frame_interval, max_frames_per_video):
                total_extracted += max_frames_per_video

        print(f"\n🎉 Пакетное извлечение завершено!")
        print(f"📊 Обработано видео: {len(video_files)}")
        print(f"📸 Всего извлечено кадров: ~{total_extracted}")

        self._generate_extraction_report()

    def extract_diverse_frames(self, video_path, oven_id=1, num_frames=200):
        """
        Извлечение разнообразных кадров из разных частей видео

        Args:
            video_path: путь к видеофайлу
            oven_id: ID печи
            num_frames: количество кадров для извлечения
        """
        if not os.path.exists(video_path):
            print(f"❌ Видеофайл не найден: {video_path}")
            return False

        print(f"🎲 Извлекаем {num_frames} разнообразных кадров из {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Генерируем равномерно распределенные позиции
        if total_frames < num_frames:
            frame_positions = list(range(0, total_frames, max(1, total_frames // num_frames)))
        else:
            step = total_frames // num_frames
            frame_positions = list(range(0, total_frames, step))[:num_frames]

        print(f"   📊 Видео: {total_frames} кадров, извлекаем из позиций: {frame_positions[:5]}...")

        extracted_count = 0
        video_name = Path(video_path).stem

        for i, frame_pos in enumerate(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            if ret:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f"diverse_{video_name}_oven{oven_id}_{i:04d}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, 'images', filename)

                cv2.imwrite(filepath, frame)

                # Метаданные
                self._save_frame_metadata(
                    filename, oven_id, video_path,
                    frame_pos, total_frames, cap.get(cv2.CAP_PROP_FPS), frame.shape
                )

                extracted_count += 1

                if extracted_count % 50 == 0:
                    progress = (i / len(frame_positions)) * 100
                    print(f"   📸 Извлечено {extracted_count}/{num_frames} кадров ({progress:.1f}%)")

        cap.release()
        print(f"✅ Извлечено {extracted_count} разнообразных кадров")
        return True

    def _extract_oven_id_from_filename(self, filename):
        """Попытка извлечь ID печи из имени файла"""
        # Ищем паттерны типа "oven1", "печь_2", "ch01" и т.д.
        import re

        patterns = [
            r'oven[_-]?(\d+)',
            r'печь[_-]?(\d+)',
            r'ch(\d+)',
            r'camera[_-]?(\d+)',
            r'cam(\d+)'
        ]

        filename_lower = filename.lower()

        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return int(match.group(1))

        # Если не нашли - возвращаем 1 по умолчанию
        return 1

    def _save_frame_metadata(self, filename, oven_id, video_path, frame_number, total_frames, fps, frame_shape):
        """Сохранение метаданных кадра"""
        metadata = {
            'filename': filename,
            'source_video': str(video_path),
            'oven_id': oven_id,
            'frame_number': frame_number,
            'total_frames': total_frames,
            'video_fps': fps,
            'timestamp': datetime.now().isoformat(),
            'resolution': {
                'width': frame_shape[1],
                'height': frame_shape[0],
                'channels': frame_shape[2]
            },
            'video_progress': frame_number / total_frames if total_frames > 0 else 0
        }

        metadata_file = os.path.join(self.output_dir, 'metadata',
                                     filename.replace('.jpg', '_metadata.json'))

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _generate_extraction_report(self):
        """Генерация отчета об извлечении данных"""
        images_dir = os.path.join(self.output_dir, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        # Статистика по источникам
        source_stats = {}
        oven_stats = {}

        metadata_dir = os.path.join(self.output_dir, 'metadata')
        if os.path.exists(metadata_dir):
            for metadata_file in os.listdir(metadata_dir):
                if metadata_file.endswith('_metadata.json'):
                    try:
                        with open(os.path.join(metadata_dir, metadata_file), 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        source = Path(metadata['source_video']).name
                        oven_id = metadata['oven_id']

                        source_stats[source] = source_stats.get(source, 0) + 1
                        oven_stats[oven_id] = oven_stats.get(oven_id, 0) + 1

                    except Exception as e:
                        print(f"⚠️  Ошибка чтения метаданных {metadata_file}: {e}")

        report = {
            'extraction_date': datetime.now().isoformat(),
            'total_extracted_frames': len(image_files),
            'frames_by_source_video': source_stats,
            'frames_by_oven': oven_stats,
            'output_directory': self.output_dir
        }

        # Сохраняем отчет
        report_path = os.path.join(self.output_dir, 'extraction_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Выводим отчет
        print("\n📊 ОТЧЕТ ОБ ИЗВЛЕЧЕНИИ ДАННЫХ")
        print("=" * 40)
        print(f"🖼️  Всего кадров: {len(image_files)}")

        if source_stats:
            print("\n📹 По видеофайлам:")
            for source, count in sorted(source_stats.items()):
                print(f"   {source}: {count} кадров")

        if oven_stats:
            print("\n🔥 По печам:")
            for oven_id, count in sorted(oven_stats.items()):
                print(f"   Печь {oven_id}: {count} кадров")

        print(f"\n📄 Отчет сохранен: {report_path}")
        print("=" * 40)


def main():
    """Главная функция для запуска извлечения"""
    import argparse

    parser = argparse.ArgumentParser(description='Извлечение кадров из MP4 записей')
    parser.add_argument('--video', help='Путь к одному видеофайлу')
    parser.add_argument('--videos-dir', help='Папка с видеофайлами')
    parser.add_argument('--output', default='training_data', help='Папка для сохранения')
    parser.add_argument('--interval', type=int, default=30, help='Интервал между кадрами')
    parser.add_argument('--max-frames', type=int, default=500, help='Максимум кадров с видео')
    parser.add_argument('--diverse', action='store_true', help='Извлекать разнообразные кадры')

    args = parser.parse_args()

    print("🎬 ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ MP4 ЗАПИСЕЙ")
    print("=" * 50)

    try:
        extractor = VideoDataExtractor(args.output)

        if args.video:
            # Обработка одного файла
            if args.diverse:
                extractor.extract_diverse_frames(args.video, num_frames=args.max_frames)
            else:
                extractor.extract_frames_from_video(args.video, frame_interval=args.interval,
                                                    max_frames=args.max_frames)

        elif args.videos_dir:
            # Пакетная обработка
            extractor.batch_extract_from_directory(args.videos_dir, args.interval, args.max_frames)

        else:
            print("❌ Укажите --video или --videos-dir")
            print("\nПримеры:")
            print("  python video_data_extractor.py --video /path/to/video.mp4")
            print("  python video_data_extractor.py --videos-dir /path/to/videos/")
            print("  python video_data_extractor.py --videos-dir ./videos --diverse --max-frames 200")

    except KeyboardInterrupt:
        print("\n🛑 Извлечение прервано пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()