# production_counter.py - Счетчик производства с интеграцией в 1C
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import time
import json
import requests
from collections import deque
from models import ProductionBatch, Product, DetectionEvent
import logging


class ProductionCounter:
    """Счетчик производства с учетом типов продукции"""

    def __init__(self, db_session, oven_id):
        self.db_session = db_session
        self.oven_id = oven_id
        self.current_batch = None
        self.current_product = None

        # Маппинг маркеров на продукты
        self.marker_to_product = self._load_marker_mapping()

        # Валидация подсчета
        self.validation_window = 60  # секунд для валидации
        self.last_counts = deque(maxlen=10)

        # Интеграция с 1C
        self.c1_config = self._load_1c_config()

        # Логирование
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ProductionCounter_Oven_{oven_id}")

        self.logger.info(f"Инициализирован счетчик для печи {oven_id}")

    def _load_marker_mapping(self):
        """Загрузка маппинга маркеров на продукты из БД"""
        products = self.db_session.query(Product).all()
        mapping = {}

        for product in products:
            if product.marker_shape:
                mapping[product.marker_shape] = product

        self.logger.info(f"Загружено {len(mapping)} маппингов маркеров")
        return mapping

    def _load_1c_config(self):
        """Загрузка конфигурации для интеграции с 1C"""
        config_file = '1c_config.json'
        default_config = {
            'enabled': False,
            'api_url': 'http://localhost:8080/api/production',
            'auth_token': '',
            'timeout': 30,
            'retry_attempts': 3
        }

        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info("Конфигурация 1C загружена")
            else:
                config = default_config
                self.logger.warning("Конфигурация 1C не найдена, используем значения по умолчанию")
        except Exception as e:
            config = default_config
            self.logger.error(f"Ошибка загрузки конфигурации 1C: {e}")

        return config

    def process_detections(self, bread_detections, marker_detections, tracked_objects, timestamp):
        """Обработка результатов детекции"""
        # Обработка маркеров - смена типа продукции
        for marker in marker_detections:
            if self.handle_marker_detection(marker, timestamp):
                break  # Обрабатываем только первый найденный маркер

        # Обработка подсчета хлеба
        if self.current_batch and tracked_objects:
            self.update_batch_count(tracked_objects, timestamp)

    def handle_marker_detection(self, marker, timestamp):
        """Обработка обнаружения маркера - смена типа продукции"""
        marker_shape = (marker.get('class_name') or '').lower()

        if marker_shape in self.marker_to_product:
            product = self.marker_to_product[marker_shape]

            # Завершаем текущую партию
            if self.current_batch:
                self.finish_current_batch()

            # Начинаем новую партию
            self.start_new_batch(product, timestamp)
            return True
        else:
            self.logger.warning(f"Неизвестный маркер: {marker_shape}")
            return False

    def start_new_batch(self, product, timestamp):
        """Начало новой партии производства"""
        try:
            self.current_batch = ProductionBatch(
                oven_id=self.oven_id,
                product_id=product.id,
                start_time=datetime.fromtimestamp(timestamp)
            )
            self.current_product = product

            self.db_session.add(self.current_batch)
            self.db_session.commit()

            self.logger.info(f"🥖 Новая партия: {product.name} на печи {self.oven_id}")

            # Записываем событие в лог
            self._log_detection_event('batch_started', timestamp, product.name)

        except Exception as e:
            self.logger.error(f"Ошибка создания новой партии: {e}")
            self.db_session.rollback()

    def update_batch_count(self, tracked_objects, timestamp):
        """Обновление счетчика партии на основе трекинга"""
        if not self.current_batch:
            return

        try:
            # Подсчитываем объекты, которые были учтены трекером
            new_bread_count = 0
            new_defect_count = 0

            for obj_id, obj_data in tracked_objects.items():
                if obj_data.get('counted', False) and obj_data.get('class_name') == 'bread':
                    new_bread_count += 1

                    # Проверяем на брак (если есть информация)
                    if obj_data.get('is_defective', False):
                        new_defect_count += 1

            # Обновляем счетчики только если есть новые объекты
            if new_bread_count > 0:
                previous_count = self.current_batch.total_count
                self.current_batch.total_count += new_bread_count
                self.current_batch.defect_count += new_defect_count

                # Обновляем среднюю уверенность трекинга
                confidences = [obj.get('confidence_history', [0.5])[-1]
                               for obj in tracked_objects.values()
                               if obj.get('counted', False)]

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    self.current_batch.tracking_confidence = avg_confidence

                self.db_session.commit()

                # Логируем событие
                self._log_detection_event('bread_counted', timestamp,
                                          f"+{new_bread_count} (всего: {self.current_batch.total_count})")

                self.logger.info(f"📊 Подсчет обновлен: +{new_bread_count} = {self.current_batch.total_count} шт "
                                 f"(брак: +{new_defect_count})")

                # Валидация счета
                self._validate_count(previous_count, new_bread_count)

        except Exception as e:
            self.logger.error(f"Ошибка обновления счетчика: {e}")
            self.db_session.rollback()

    def _validate_count(self, previous_count, increment):
        """Валидация правильности подсчета"""
        current_count = self.current_batch.total_count

        # Проверяем на аномальные скачки
        if increment > 20:  # Более 20 объектов за раз - подозрительно
            self.logger.warning(f"⚠️  Большой скачок в подсчете: +{increment}")

        # Сохраняем историю для анализа тенденций
        self.last_counts.append({
            'timestamp': time.time(),
            'count': current_count,
            'increment': increment
        })

        # Проверяем стабильность подсчета
        if len(self.last_counts) >= 5:
            recent_increments = [c['increment'] for c in list(self.last_counts)[-5:]]
            avg_increment = sum(recent_increments) / len(recent_increments)

            if increment > avg_increment * 3:  # Текущий прирост в 3 раза больше среднего
                self.logger.warning(f"⚠️  Подозрительный прирост: {increment} vs среднее {avg_increment:.1f}")

    def finish_current_batch(self):
        """Завершение текущей партии"""
        if not self.current_batch:
            return

        try:
            self.current_batch.end_time = datetime.now()

            # Финальная валидация счета
            final_count = self._validate_final_count()
            self.current_batch.total_count = final_count

            self.db_session.commit()

            # Логируем завершение партии
            self._log_detection_event('batch_finished', time.time(),
                                      f"Итого: {final_count} шт, брак: {self.current_batch.defect_count}")

            # Отправляем данные в 1C
            if self.c1_config.get('enabled', False):
                self.send_to_1c(self.current_batch)

            self.logger.info(f"✅ Партия завершена: {final_count} шт, "
                             f"продукт: {self.current_product.name}, "
                             f"брак: {self.current_batch.defect_count}")

            # Очищаем текущие данные
            self.current_batch = None
            self.current_product = None

        except Exception as e:
            self.logger.error(f"Ошибка завершения партии: {e}")
            self.db_session.rollback()

    def _validate_final_count(self):
        """Финальная валидация количества перед отправкой в 1C"""
        if not self.current_batch:
            return 0

        current_count = self.current_batch.total_count

        # Проверяем разумность итогового количества
        batch_duration = (datetime.now() - self.current_batch.start_time).total_seconds() / 60  # минуты

        if batch_duration > 0:
            items_per_minute = current_count / batch_duration

            # Если больше 50 штук в минуту - подозрительно
            if items_per_minute > 50:
                self.logger.warning(f"⚠️  Высокая скорость производства: {items_per_minute:.1f} шт/мин")

            # Если меньше 1 штуки в 5 минут - тоже странно
            if items_per_minute < 0.2 and current_count > 0:
                self.logger.warning(f"⚠️  Очень низкая скорость производства: {items_per_minute:.1f} шт/мин")

        return current_count

    def _log_detection_event(self, event_type, timestamp, details):
        """Логирование события детекции в БД"""
        try:
            if self.current_batch:
                event = DetectionEvent(
                    batch_id=self.current_batch.id,
                    timestamp=datetime.fromtimestamp(timestamp),
                    event_type=event_type,
                    confidence=0.95,  # Базовая уверенность
                    bbox_data=json.dumps({'details': details})
                )
                self.db_session.add(event)
                self.db_session.commit()
        except Exception as e:
            self.logger.error(f"Ошибка логирования события: {e}")

    def send_to_1c(self, batch):
        """Отправка данных в 1C через REST API"""
        if not self.c1_config.get('enabled', False):
            self.logger.info("Интеграция с 1C отключена")
            return False

        try:
            # Подготавливаем данные для отправки
            data = {
                'oven_id': batch.oven_id,
                'product_code': batch.product.code,
                'product_name': batch.product.name,
                'quantity': batch.total_count,
                'defects': batch.defect_count,
                'start_time': batch.start_time.isoformat(),
                'end_time': batch.end_time.isoformat() if batch.end_time else None,
                'tracking_confidence': batch.tracking_confidence,
                'batch_id': batch.id
            }

            # Заголовки запроса
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.c1_config.get("auth_token", "")}'
            }

            # Отправляем данные
            url = self.c1_config.get('api_url')
            timeout = self.c1_config.get('timeout', 30)

            self.logger.info(f"📤 Отправка в 1C: {batch.total_count} шт {batch.product.name}")

            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout
            )

            if response.status_code == 200:
                self.logger.info("✅ Данные успешно отправлены в 1C")
                return True
            else:
                self.logger.error(f"❌ Ошибка отправки в 1C: {response.status_code} - {response.text}")
                return False

        except requests.exceptions.Timeout:
            self.logger.error("❌ Таймаут при отправке в 1C")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error("❌ Ошибка подключения к 1C")
            return False
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки в 1C: {e}")
            return False

    def get_current_batch_info(self):
        """Получение информации о текущей партии"""
        if not self.current_batch:
            return None

        duration = (datetime.now() - self.current_batch.start_time).total_seconds() / 60

        return {
            'batch_id': self.current_batch.id,
            'product_name': self.current_product.name if self.current_product else 'Unknown',
            'product_code': self.current_product.code if self.current_product else 'N/A',
            'start_time': self.current_batch.start_time.isoformat(),
            'duration_minutes': round(duration, 1),
            'total_count': self.current_batch.total_count,
            'defect_count': self.current_batch.defect_count,
            'tracking_confidence': self.current_batch.tracking_confidence or 0.0
        }

    def get_statistics(self, hours=24):
        """Получение статистики производства за последние N часов"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)

            # Запрос завершенных партий
            batches = self.db_session.query(ProductionBatch).filter(
                ProductionBatch.oven_id == self.oven_id,
                ProductionBatch.start_time >= start_time,
                ProductionBatch.end_time != None
            ).all()

            if not batches:
                return {
                    'total_batches': 0,
                    'total_items': 0,
                    'total_defects': 0,
                    'products': {}
                }

            # Подсчитываем статистику
            total_items = sum(batch.total_count for batch in batches)
            total_defects = sum(batch.defect_count for batch in batches)

            # Статистика по продуктам
            products_stats = {}
            for batch in batches:
                product_name = batch.product.name
                if product_name not in products_stats:
                    products_stats[product_name] = {
                        'batches': 0,
                        'items': 0,
                        'defects': 0
                    }

                products_stats[product_name]['batches'] += 1
                products_stats[product_name]['items'] += batch.total_count
                products_stats[product_name]['defects'] += batch.defect_count

            return {
                'total_batches': len(batches),
                'total_items': total_items,
                'total_defects': total_defects,
                'defect_rate': (total_defects / total_items * 100) if total_items > 0 else 0,
                'products': products_stats,
                'period_hours': hours
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return None

    def force_finish_batch(self, reason="Manual finish"):
        """Принудительное завершение текущей партии"""
        if self.current_batch:
            self.logger.info(f"🛑 Принудительное завершение партии: {reason}")
            self._log_detection_event('batch_force_finished', time.time(), reason)
            self.finish_current_batch()
            return True
        return False

    def reset_batch_count(self):
        """Сброс счетчика текущей партии (для исправления ошибок)"""
        if self.current_batch:
            old_count = self.current_batch.total_count
            self.current_batch.total_count = 0
            self.current_batch.defect_count = 0
            self.db_session.commit()

            self.logger.warning(f"🔄 Счетчик партии сброшен: {old_count} -> 0")
            self._log_detection_event('batch_count_reset', time.time(), f"Сброшен счетчик: {old_count}")
            return True
        return False