# smart_production_counter.py - Умный счетчик с автоопределением партий
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import time
import json
import requests
from collections import deque
from models import ProductionBatch, Product, DetectionEvent
from smart_batch_detector import SmartBatchDetector
import logging


class SmartProductionCounter:
    """Умный счетчик производства с автоопределением смены партий"""

    def __init__(self, db_session, oven_id):
        self.db_session = db_session
        self.oven_id = oven_id

        # Умный детектор партий
        self.batch_detector = SmartBatchDetector(oven_id)

        # Текущая партия
        self.current_batch = None
        self.current_product = None

        # Маппинг типов хлеба на продукты в БД
        self.bread_type_to_product = self._load_product_mapping()

        # Валидация подсчета
        self.validation_window = 60
        self.last_counts = deque(maxlen=10)

        # Интеграция с 1C
        self.c1_config = self._load_1c_config()

        # Логирование
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"SmartProductionCounter_Oven_{oven_id}")

        self.logger.info(f"🧠 Инициализирован умный счетчик для печи {oven_id}")

    def _load_product_mapping(self):
        """Загрузка маппинга типов хлеба на продукты из БД"""
        mapping = {}

        # Создаем базовые продукты если их нет
        default_products = {
            'white_bread': 'Белый хлеб',
            'dark_bread': 'Черный хлеб',
            'baton': 'Батон',
            'molded_bread': 'Хлеб в формах'
        }

        for bread_type, product_name in default_products.items():
            # Ищем продукт в БД
            product = self.db_session.query(Product).filter_by(name=product_name).first()

            if not product:
                # Создаем новый продукт
                product = Product(
                    name=product_name,
                    code=f"{bread_type.upper()[:3]}001",
                    marker_shape=None  # Больше не используем маркеры
                )
                self.db_session.add(product)
                self.db_session.commit()
                self.logger.info(f"✅ Создан продукт: {product_name}")

            mapping[bread_type] = product

        self.logger.info(f"📋 Загружено {len(mapping)} маппингов продуктов")
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
            import os
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

    def process_detections(self, bread_detections, tracked_objects, frame, timestamp):
        """Обработка результатов детекции с умным определением партий"""

        # 1. Анализ смены партий
        batch_info = self.batch_detector.process_frame(bread_detections, frame, timestamp)

        # 2. Обработка смены партии
        if batch_info['batch_type'] != (
        self.current_product.code.split('001')[0].lower() + '_bread' if self.current_product else None):
            self._handle_batch_change(batch_info, timestamp)

        # 3. Обновление счетчика только если есть активная партия
        if self.current_batch and tracked_objects and not batch_info['in_gap']:
            self._update_batch_count(tracked_objects, timestamp)

        # 4. Логирование состояния
        if batch_info['in_gap']:
            self.logger.debug(f"⏸️  Пропуск, подсчет приостановлен")
        elif batch_info['waiting_for_new_batch']:
            self.logger.debug(f"🔄 Ожидание новой партии")

    def _handle_batch_change(self, batch_info, timestamp):
        """Обработка смены партии"""
        batch_type = batch_info['batch_type']

        if not batch_type:
            return

        # Завершаем текущую партию
        if self.current_batch:
            self.finish_current_batch()

        # Начинаем новую партию
        if batch_type in self.bread_type_to_product:
            product = self.bread_type_to_product[batch_type]
            self.start_new_batch(product, timestamp)

            # Логируем событие смены
            self._log_detection_event(
                'batch_auto_changed',
                timestamp,
                f"Автосмена: {batch_info['batch_name']}"
            )

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

    def _update_batch_count(self, tracked_objects, timestamp):
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
                'batch_id': batch.id,
                'detection_method': 'auto_visual_classification'  # Указываем метод
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

        # Получаем информацию от умного детектора
        smart_info = self.batch_detector.get_current_batch_info()

        return {
            'batch_id': self.current_batch.id,
            'product_name': self.current_product.name if self.current_product else 'Unknown',
            'product_code': self.current_product.code if self.current_product else 'N/A',
            'start_time': self.current_batch.start_time.isoformat(),
            'duration_minutes': round(duration, 1),
            'total_count': self.current_batch.total_count,
            'defect_count': self.current_batch.defect_count,
            'tracking_confidence': self.current_batch.tracking_confidence or 0.0,
            'auto_detected_type': smart_info['name'],
            'in_gap': smart_info['in_gap'],
            'waiting_for_new_batch': smart_info['waiting_for_new_batch']
        }

    def force_batch_change(self, new_product_type=None):
        """Принудительная смена партии"""
        if new_product_type and new_product_type in self.bread_type_to_product:
            # Завершаем текущую партию
            if self.current_batch:
                self.finish_current_batch()

            # Начинаем новую с указанным типом
            product = self.bread_type_to_product[new_product_type]
            self.start_new_batch(product, time.time())

            # Принудительно устанавливаем тип в детекторе
            self.batch_detector.force_new_batch(new_product_type)

            self.logger.info(f"🔄 Принудительная смена на: {product.name}")
        else:
            # Просто сбрасываем для автоопределения
            if self.current_batch:
                self.finish_current_batch()

            self.batch_detector.force_new_batch()
            self.logger.info(f"🔄 Сброс для автоопределения следующей партии")

    def get_smart_statistics(self):
        """Расширенная статистика с умной аналитикой"""
        base_stats = self.get_statistics()
        smart_stats = self.batch_detector.get_statistics()

        return {
            **base_stats,
            'smart_detection': smart_stats,
            'auto_batch_changes': smart_stats.get('type_changes_detected', 0),
            'gap_detection_accuracy': smart_stats.get('gap_detection_ratio', 0)
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
                'period_hours': hours,
                'detection_method': 'auto_visual_classification'
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return None