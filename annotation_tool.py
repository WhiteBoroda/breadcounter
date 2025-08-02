# annotation_tool.py - Инструмент для разметки данных
import cv2
import json
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float = 1.0


class AnnotationTool:
    """Инструмент для разметки обучающих данных"""

    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.current_image = None
        self.current_annotations = []
        self.drawing = False
        self.start_point = None
        self.current_class = 'bread'

        # Создаем папку для аннотаций
        os.makedirs(annotations_dir, exist_ok=True)

        # Классы для аннотации
        self.classes = ['bread', 'circle', 'square', 'triangle', 'diamond', 'star', 'defective_bread']
        self.class_colors = {
            'bread': (0, 255, 0),  # Зеленый
            'circle': (255, 0, 0),  # Синий
            'square': (0, 0, 255),  # Красный
            'triangle': (255, 255, 0),  # Голубой
            'diamond': (255, 0, 255),  # Фиолетовый
            'star': (0, 255, 255),  # Желтый
            'defective_bread': (0, 128, 255)  # Оранжевый
        }

        print("🏷️  Инструмент аннотации инициализирован")
        print(f"📁 Изображения: {images_dir}")
        print(f"📁 Аннотации: {annotations_dir}")
        print(f"🎨 Доступные классы: {', '.join(self.classes)}")

    def annotate_dataset(self):
        """Основной процесс аннотации"""
        image_files = [f for f in os.listdir(self.images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print("❌ Изображения не найдены!")
            return

        image_files.sort()

        print("\n🏷️  ИНСТРУМЕНТ АННОТАЦИИ")
        print("=" * 50)
        self._print_instructions()
        print(f"📸 Найдено {len(image_files)} изображений для разметки")

        current_idx = 0

        while current_idx < len(image_files):
            image_file = image_files[current_idx]
            print(f"\n📷 Обрабатываем: {image_file} ({current_idx + 1}/{len(image_files)})")

            # Загружаем изображение
            image_path = os.path.join(self.images_dir, image_file)
            self.current_image = cv2.imread(image_path)

            if self.current_image is None:
                print(f"❌ Не удалось загрузить {image_file}")
                current_idx += 1
                continue

            # Загружаем существующие аннотации
            self._load_annotations(image_file)

            # Настраиваем обработчики мыши
            cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Annotation Tool', self._mouse_callback)

            # Главный цикл для текущего изображения
            while True:
                display_image = self._draw_annotations()
                cv2.imshow('Annotation Tool', display_image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('n'):  # Следующее изображение
                    current_idx += 1
                    break
                elif key == ord('p'):  # Предыдущее изображение
                    current_idx = max(0, current_idx - 1)
                    break
                elif key == ord('s'):  # Сохранить
                    self._save_annotations(image_file)
                    print(f"💾 Аннотации сохранены для {image_file}")
                elif key == ord('d'):  # Удалить последнюю
                    if self.current_annotations:
                        deleted = self.current_annotations.pop()
                        print(f"🗑️  Удалена аннотация: {deleted.class_name}")
                elif key == ord('c'):  # Очистить все
                    self.current_annotations.clear()
                    print("🗑️  Все аннотации очищены")
                elif key == ord('a'):  # Автосохранение и переход к следующему
                    self._save_annotations(image_file)
                    current_idx += 1
                    break
                elif key == ord('h'):  # Помощь
                    self._print_instructions()
                elif key == ord('q'):  # Выход
                    self._save_annotations(image_file)  # Сохраняем перед выходом
                    cv2.destroyAllWindows()
                    return
                elif ord('1') <= key <= ord('7'):  # Выбор класса
                    class_idx = key - ord('1')
                    if class_idx < len(self.classes):
                        self.current_class = self.classes[class_idx]
                        print(f"🏷️  Выбран класс: {self.current_class}")

        cv2.destroyAllWindows()
        print("\n✅ Аннотация завершена!")
        self._print_annotation_summary()

    def _print_instructions(self):
        """Вывод инструкций по использованию"""
        print("\n📋 УПРАВЛЕНИЕ:")
        print("  🖱️  Левая кнопка мыши: Рисовать прямоугольник")
        print("  [1-7] Выбор класса:")
        for i, class_name in enumerate(self.classes):
            color_name = list(self.class_colors.keys())[i]
            print(f"    [{i + 1}] {class_name}")
        print("\n  [n] Следующее изображение")
        print("  [p] Предыдущее изображение")
        print("  [s] Сохранить аннотации")
        print("  [a] Автосохранение + следующее")
        print("  [d] Удалить последнюю аннотацию")
        print("  [c] Очистить все аннотации")
        print("  [h] Показать справку")
        print("  [q] Выход")
        print("=" * 50)

    def _mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                # Создаем bounding box
                x1, y1 = self.start_point
                x2, y2 = x, y

                # Убеждаемся что координаты правильные
                bbox = BoundingBox(
                    x1=min(x1, x2),
                    y1=min(y1, y2),
                    x2=max(x1, x2),
                    y2=max(y1, y2),
                    class_name=self.current_class
                )

                # Проверяем минимальный размер
                width = bbox.x2 - bbox.x1
                height = bbox.y2 - bbox.y1

                if width > 10 and height > 10:
                    self.current_annotations.append(bbox)
                    print(f"➕ Добавлена аннотация: {self.current_class} ({width}x{height})")
                else:
                    print("⚠️  Слишком маленький прямоугольник, аннотация не добавлена")

                self.drawing = False
                self.start_point = None

        elif event == cv2.EVENT_MOUSEMOVE:
            # Показываем текущий прямоугольник при рисовании
            if self.drawing and self.start_point:
                pass  # Обновление происходит в _draw_annotations

    def _draw_annotations(self):
        """Отрисовка аннотаций на изображении"""
        display_image = self.current_image.copy()

        # Отрисовываем существующие аннотации
        for bbox in self.current_annotations:
            color = self.class_colors.get(bbox.class_name, (128, 128, 128))

            # Прямоугольник
            cv2.rectangle(display_image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

            # Подпись класса
            label = f"{bbox.class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Фон для текста
            cv2.rectangle(display_image,
                          (bbox.x1, bbox.y1 - label_size[1] - 8),
                          (bbox.x1 + label_size[0] + 4, bbox.y1),
                          color, -1)

            # Текст
            cv2.putText(display_image, label,
                        (bbox.x1 + 2, bbox.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Показываем текущий рисуемый прямоугольник
        if self.drawing and self.start_point:
            mouse_pos = cv2.getWindowProperty('Annotation Tool', cv2.WND_PROP_AUTOSIZE)
            # Для простоты не показываем временный прямоугольник

        # Информационная панель
        self._draw_info_panel(display_image)

        return display_image

    def _draw_info_panel(self, image):
        """Отрисовка информационной панели"""
        h, w = image.shape[:2]

        # Фон для панели
        panel_height = 120
        cv2.rectangle(image, (0, h - panel_height), (w, h), (0, 0, 0), -1)

        # Текущий класс
        current_color = self.class_colors.get(self.current_class, (255, 255, 255))
        cv2.putText(image, f"Current class: {self.current_class}",
                    (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

        # Количество аннотаций
        cv2.putText(image, f"Annotations: {len(self.current_annotations)}",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Статистика по классам
        class_counts = {}
        for bbox in self.current_annotations:
            class_counts[bbox.class_name] = class_counts.get(bbox.class_name, 0) + 1

        stats_text = " | ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        if stats_text:
            cv2.putText(image, stats_text,
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Подсказка
        cv2.putText(image, "Press [h] for help",
                    (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def _load_annotations(self, image_file):
        """Загрузка существующих аннотаций"""
        annotation_file = os.path.splitext(image_file)[0] + '.json'
        annotation_path = os.path.join(self.annotations_dir, annotation_file)

        self.current_annotations = []

        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for ann in data.get('annotations', []):
                    bbox = BoundingBox(
                        x1=ann['x1'], y1=ann['y1'],
                        x2=ann['x2'], y2=ann['y2'],
                        class_name=ann['class_name'],
                        confidence=ann.get('confidence', 1.0)
                    )
                    self.current_annotations.append(bbox)

                print(f"📄 Загружено {len(self.current_annotations)} существующих аннотаций")

            except Exception as e:
                print(f"⚠️  Ошибка загрузки аннотаций: {e}")

    def _save_annotations(self, image_file):
        """Сохранение аннотаций"""
        if not self.current_annotations:
            return

        annotation_file = os.path.splitext(image_file)[0] + '.json'
        annotation_path = os.path.join(self.annotations_dir, annotation_file)

        data = {
            'image_file': image_file,
            'image_size': {
                'width': self.current_image.shape[1],
                'height': self.current_image.shape[0],
                'channels': self.current_image.shape[2]
            },
            'annotations': []
        }

        for bbox in self.current_annotations:
            annotation = {
                'x1': bbox.x1, 'y1': bbox.y1,
                'x2': bbox.x2, 'y2': bbox.y2,
                'class_name': bbox.class_name,
                'confidence': bbox.confidence,
                'width': bbox.x2 - bbox.x1,
                'height': bbox.y2 - bbox.y1,
                'area': (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            }
            data['annotations'].append(annotation)

        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")

    def _print_annotation_summary(self):
        """Вывод сводной статистики аннотации"""
        annotation_files = [f for f in os.listdir(self.annotations_dir)
                            if f.endswith('.json')]

        if not annotation_files:
            print("📊 Аннотации не найдены")
            return

        total_annotations = 0
        class_stats = {}

        for ann_file in annotation_files:
            ann_path = os.path.join(self.annotations_dir, ann_file)
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for ann in data.get('annotations', []):
                    class_name = ann['class_name']
                    class_stats[class_name] = class_stats.get(class_name, 0) + 1
                    total_annotations += 1

            except Exception as e:
                print(f"⚠️  Ошибка чтения {ann_file}: {e}")

        print("\n📊 СТАТИСТИКА АННОТАЦИИ")
        print("=" * 40)
        print(f"🖼️  Файлов с аннотациями: {len(annotation_files)}")
        print(f"🏷️  Всего аннотаций: {total_annotations}")
        print("\n📈 По классам:")

        for class_name, count in sorted(class_stats.items()):
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")

        print("=" * 40)


def main():
    """Главная функция для запуска инструмента аннотации"""
    import argparse

    parser = argparse.ArgumentParser(description='Инструмент аннотации для обучения модели')
    parser.add_argument('--images', default='training_data/images',
                        help='Папка с изображениями')
    parser.add_argument('--annotations', default='training_data/annotations',
                        help='Папка для сохранения аннотаций')

    args = parser.parse_args()

    if not os.path.exists(args.images):
        print(f"❌ Папка с изображениями не найдена: {args.images}")
        print("   Сначала запустите сбор данных: python training_pipeline.py")
        return

    print("🏷️  ИНСТРУМЕНТ АННОТАЦИИ ДАННЫХ")
    print("=" * 50)
    print(f"📁 Изображения: {args.images}")
    print(f"📁 Аннотации: {args.annotations}")

    try:
        annotator = AnnotationTool(args.images, args.annotations)
        annotator.annotate_dataset()

    except KeyboardInterrupt:
        print("\n🛑 Аннотация прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()