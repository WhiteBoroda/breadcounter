# model_trainer.py - Обучение модели детекции хлеба
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path


class BreadDetectionTrainer:
    """Обучение модели детекции хлеба для Coral TPU"""

    def __init__(self, data_dir, model_name='bread_detector'):
        self.data_dir = data_dir
        self.model_name = model_name
        self.classes = ['background', 'bread', 'circle', 'square', 'triangle', 'diamond', 'star', 'defective_bread']
        self.num_classes = len(self.classes)
        self.input_size = (320, 320)  # Оптимальный размер для мобильных моделей

        print(f"🎓 Инициализация тренера модели")
        print(f"📁 Данные: {data_dir}")
        print(f"🏷️  Классы: {', '.join(self.classes[1:])}")  # Без background

    def prepare_dataset(self):
        """Подготовка датасета из аннотированных данных"""
        print("📊 Подготовка датасета...")

        images_dir = os.path.join(self.data_dir, 'images')
        annotations_dir = os.path.join(self.data_dir, 'annotations')

        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            raise FileNotFoundError("Папки с данными не найдены. Запустите сначала сбор и аннотацию данных.")

        images = []
        labels = []

        # Загружаем все аннотированные изображения
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]

        print(f"📄 Найдено аннотированных файлов: {len(annotation_files)}")

        for ann_file in annotation_files:
            ann_path = os.path.join(annotations_dir, ann_file)
            img_file = ann_file.replace('.json', '.jpg')
            img_path = os.path.join(images_dir, img_file)

            if not os.path.exists(img_path):
                print(f"⚠️  Пропускаем {img_file} - изображение не найдено")
                continue

            # Загружаем изображение
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  Не удалось загрузить {img_file}")
                continue

            # Загружаем аннотации
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)

                annotations = ann_data.get('annotations', [])
                if not annotations:
                    print(f"⚠️  Нет аннотаций в {ann_file}")
                    continue

                images.append(image)
                labels.append(annotations)

            except Exception as e:
                print(f"❌ Ошибка загрузки аннотаций {ann_file}: {e}")
                continue

        if len(images) < 5:
            raise ValueError(f"Недостаточно данных для обучения. Найдено только {len(images)} изображений. Минимум 5.")

        print(f"✅ Загружено {len(images)} изображений с аннотациями")

        # Разделяем на train/val
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        print(f"🎓 Обучающий набор: {len(train_images)}")
        print(f"✅ Валидационный набор: {len(val_images)}")

        return (train_images, train_labels), (val_images, val_labels)

    def create_model(self):
        """Создание модели на основе MobileNetV2"""
        print("🏗️  Создание модели...")

        # Базовая модель MobileNetV2 (оптимизированная для Coral TPU)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.input_size[0], self.input_size[1], 3),
            include_top=False,
            weights='imagenet'
        )

        # Заморозим базовую модель для transfer learning
        base_model.trainable = False

        # Добавляем голову для детекции объектов
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        print("📋 Архитектура модели:")
        model.summary()

        return model

    def train_model(self, epochs=50, batch_size=16):
        """Обучение модели"""
        print("🎓 Начинаем обучение модели...")

        # Подготавливаем данные
        try:
            (train_images, train_labels), (val_images, val_labels) = self.prepare_dataset()
        except Exception as e:
            print(f"❌ Ошибка подготовки данных: {e}")
            return None, None

        # Создаем модель
        model = self.create_model()

        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Подготавливаем генераторы данных
        print("🔄 Подготовка генераторов данных...")
        train_gen = self._create_data_generator(train_images, train_labels, batch_size, augment=True)
        val_gen = self._create_data_generator(val_images, val_labels, batch_size, augment=False)

        # Вычисляем количество шагов
        steps_per_epoch = max(1, len(train_images) // batch_size)
        validation_steps = max(1, len(val_images) // batch_size)

        print(f"📊 Шагов на эпоху: {steps_per_epoch}")
        print(f"📊 Валидационных шагов: {validation_steps}")

        # Callbacks для обучения
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=8,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{self.model_name}_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]

        # Обучение
        print(f"🚀 Запуск обучения на {epochs} эпох...")
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Сохраняем финальную модель
        final_model_path = f'{self.model_name}_final.h5'
        model.save(final_model_path)
        print(f"💾 Модель сохранена: {final_model_path}")

        # Сохраняем историю обучения
        self._save_training_history(history)

        # Создаем графики обучения
        self._plot_training_history(history)

        print("✅ Обучение завершено успешно!")
        return model, history

    def _create_data_generator(self, images, labels, batch_size, augment=False):
        """Создание генератора данных"""

        def generator():
            while True:
                # Перемешиваем данные
                indices = np.random.permutation(len(images))

                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_images = []
                    batch_labels = []

                    for idx in batch_indices:
                        if idx >= len(images):
                            continue

                        image = images[idx]
                        anns = labels[idx]

                        # Предобработка изображения
                        processed_image = self._preprocess_image(image, augment)

                        # Для упрощения берем класс первой аннотации
                        # В более сложной системе нужно обрабатывать все объекты
                        if anns:
                            class_name = anns[0]['class_name']
                            label = self._get_class_index(class_name)
                        else:
                            label = 0  # background

                        batch_images.append(processed_image)
                        batch_labels.append(label)

                    if batch_images:  # Проверяем что батч не пустой
                        yield np.array(batch_images), np.array(batch_labels)

        return generator()

    def _preprocess_image(self, image, augment=False):
        """Предобработка изображения"""
        # Изменяем размер
        resized = cv2.resize(image, self.input_size)

        # Аугментация данных
        if augment:
            resized = self._augment_image(resized)

        # Нормализация
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def _augment_image(self, image):
        """Аугментация изображения"""
        # Случайный поворот
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))

        # Случайное изменение яркости
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        # Случайное горизонтальное отражение
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)

        return image

    def _get_class_index(self, class_name):
        """Получение индекса класса"""
        try:
            return self.classes.index(class_name)
        except ValueError:
            print(f"⚠️  Неизвестный класс: {class_name}, используем background")
            return 0

    def _save_training_history(self, history):
        """Сохранение истории обучения"""
        history_data = {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'],
            'epochs': len(history.history['loss'])
        }

        history_path = f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"📈 История обучения сохранена: {history_path}")

    def _plot_training_history(self, history):
        """Создание графиков обучения"""
        try:
            plt.figure(figsize=(12, 4))

            # График точности
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # График потерь
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()

            plot_path = f'{self.model_name}_training_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Графики обучения сохранены: {plot_path}")

        except Exception as e:
            print(f"⚠️  Не удалось создать графики: {e}")

    def convert_to_tflite(self, model_path):
        """Конвертация в TensorFlow Lite"""
        print("🔄 Конвертация в TensorFlow Lite...")

        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            return None

        try:
            # Загружаем модель
            model = tf.keras.models.load_model(model_path)

            # Конвертируем в TF Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Настройки оптимизации
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Квантизация для Edge TPU
            converter.representative_dataset = self._representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            print("⚙️  Применяем квантизацию INT8...")
            tflite_model = converter.convert()

            # Сохраняем TF Lite модель
            tflite_path = f'{self.model_name}.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            # Информация о размере
            model_size_mb = len(tflite_model) / (1024 * 1024)
            print(f"💾 TF Lite модель сохранена: {tflite_path}")
            print(f"📏 Размер модели: {model_size_mb:.2f} MB")

            return tflite_path

        except Exception as e:
            print(f"❌ Ошибка конвертации: {e}")
            return None

    def _representative_dataset(self):
        """Представительный датасет для квантизации"""
        try:
            (train_images, _), _ = self.prepare_dataset()

            # Используем первые 100 изображений для квантизации
            sample_count = min(100, len(train_images))

            for i in range(sample_count):
                image = self._preprocess_image(train_images[i], augment=False)
                yield [np.expand_dims(image, axis=0)]

        except Exception as e:
            print(f"⚠️  Ошибка в representative dataset: {e}")
            # Возвращаем случайные данные как fallback
            for _ in range(10):
                yield [np.random.random((1, self.input_size[0], self.input_size[1], 3)).astype(np.float32)]

    def compile_for_edge_tpu(self, tflite_path):
        """Компиляция для Edge TPU"""
        print("🧠 Компиляция для Edge TPU...")

        if not os.path.exists(tflite_path):
            print(f"❌ TF Lite модель не найдена: {tflite_path}")
            return None

        output_path = f'{self.model_name}_edgetpu.tflite'

        try:
            import subprocess

            # Команда компиляции
            cmd = [
                'edgetpu_compiler',
                tflite_path,
                '-o', os.path.dirname(output_path) or '.'
            ]

            print(f"🔧 Выполняем команду: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"✅ Edge TPU модель создана: {output_path}")
                print("📊 Статистика компиляции:")
                print(result.stdout)
                return output_path
            else:
                print(f"❌ Ошибка компиляции Edge TPU:")
                print(result.stderr)
                return None

        except subprocess.TimeoutExpired:
            print("❌ Таймаут компиляции Edge TPU")
            return None
        except FileNotFoundError:
            print("❌ Edge TPU Compiler не установлен")
            print("💡 Установите: sudo apt install edgetpu-compiler")
            print("   или скачайте с https://coral.ai/software/#edgetpu-compiler")
            return None
        except Exception as e:
            print(f"❌ Ошибка компиляции: {e}")
            return None

    def create_labels_file(self):
        """Создание файла меток для модели"""
        labels_path = 'labels.txt'
        with open(labels_path, 'w', encoding='utf-8') as f:
            for i, class_name in enumerate(self.classes):
                f.write(f"{i} {class_name}\n")

        print(f"🏷️  Файл меток создан: {labels_path}")
        return labels_path


def main():
    """Главная функция для запуска обучения"""
    import argparse

    parser = argparse.ArgumentParser(description='Обучение модели детекции хлеба')
    parser.add_argument('--data', default='training_data', help='Папка с данными')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=16, help='Размер батча')

    args = parser.parse_args()

    print("🧠 ОБУЧЕНИЕ МОДЕЛИ ДЕТЕКЦИИ ХЛЕБА")
    print("=" * 50)

    try:
        # Проверяем наличие данных
        if not os.path.exists(args.data):
            print(f"❌ Папка с данными не найдена: {args.data}")
            print("   Сначала запустите сбор данных: python training_pipeline.py")
            return

        # Создаем тренер
        trainer = BreadDetectionTrainer(args.data)

        # Обучаем модель
        model, history = trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)

        if model and history:
            # Создаем файл меток
            trainer.create_labels_file()

            # Конвертируем в TF Lite
            print("\n🔄 Конвертация модели...")
            tflite_path = trainer.convert_to_tflite('bread_detector_final.h5')

            if tflite_path:
                # Компилируем для Edge TPU
                print("\n🧠 Компиляция для Coral TPU...")
                edge_tpu_path = trainer.compile_for_edge_tpu(tflite_path)

                print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
                print("=" * 50)
                print(f"📄 Обычная модель: bread_detector_final.h5")
                print(f"📱 TF Lite модель: {tflite_path}")
                if edge_tpu_path:
                    print(f"🧠 Coral TPU модель: {edge_tpu_path}")
                print(f"🏷️  Файл меток: labels.txt")
                print("\n🚀 Теперь можно запустить систему:")
                print("   python main_multicamera.py cameras.yaml")
            else:
                print("⚠️  Конвертация не удалась, но основная модель обучена")
        else:
            print("❌ Обучение не удалось")

    except KeyboardInterrupt:
        print("\n🛑 Обучение прервано пользователем")
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()