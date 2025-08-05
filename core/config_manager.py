# core/config_manager.py
"""Централизованное управление конфигурацией"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Менеджер конфигурации с валидацией и резервным копированием"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def load_cameras_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации камер с валидацией"""
        config_file = self.config_dir / "cameras.yaml"

        if not config_file.exists():
            return self._create_default_cameras_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Валидация структуры
            if not self._validate_cameras_config(config):
                self.logger.warning("Конфигурация камер не прошла валидацию")
                return self._create_default_cameras_config()

            return config

        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации камер: {e}")
            return self._create_default_cameras_config()

    def save_cameras_config(self, config: Dict[str, Any]) -> bool:
        """Сохранение конфигурации с резервной копией"""
        config_file = self.config_dir / "cameras.yaml"
        backup_file = self.config_dir / "cameras_backup.yaml"

        try:
            # Создаем резервную копию если файл существует
            if config_file.exists():
                config_file.rename(backup_file)

            # Валидируем перед сохранением
            if not self._validate_cameras_config(config):
                raise ValueError("Конфигурация не прошла валидацию")

            # Сохраняем
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f,
                          default_flow_style=False,
                          allow_unicode=True,
                          indent=2,
                          sort_keys=False)

            self.logger.info(f"Конфигурация сохранена: {len(config.get('cameras', []))} камер")

            # Удаляем резервную копию при успешном сохранении
            if backup_file.exists():
                backup_file.unlink()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {e}")

            # Восстанавливаем из резервной копии
            if backup_file.exists():
                backup_file.rename(config_file)

            return False

    def _validate_cameras_config(self, config: Dict[str, Any]) -> bool:
        """Валидация структуры конфигурации"""
        if not isinstance(config, dict):
            return False

        # Проверяем обязательные секции
        required_sections = ['cameras', 'system', 'classes']
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Отсутствует секция: {section}")
                return False

        # Валидируем камеры
        cameras = config.get('cameras', [])
        if not isinstance(cameras, list):
            return False

        for i, camera in enumerate(cameras):
            if not self._validate_camera(camera, i):
                return False

        return True

    def _validate_camera(self, camera: Dict[str, Any], index: int) -> bool:
        """Валидация отдельной камеры"""
        required_fields = ['camera_ip', 'login', 'password', 'oven_name']

        for field in required_fields:
            if field not in camera or not camera[field]:
                self.logger.error(f"Камера {index}: отсутствует поле {field}")
                return False

        # Валидация IP адреса
        ip = camera['camera_ip']
        ip_parts = ip.split('.')
        if len(ip_parts) != 4:
            return False

        try:
            for part in ip_parts:
                if not (0 <= int(part) <= 255):
                    return False
        except ValueError:
            return False

        return True

    def _create_default_cameras_config(self) -> Dict[str, Any]:
        """Создание конфигурации по умолчанию"""
        default_config = {
            'cameras': [],
            'system': {
                'tpu_devices': 1,
                'frame_rate': 15,
                'detection_threshold': 0.5,
                'tracking_max_distance': 100
            },
            'data_collection': {
                'output_dir': "training_data",
                'save_interval': 5,
                'video_duration': 30
            },
            'classes': [
                {'name': 'bread', 'color': [0, 255, 0]},
                {'name': 'bun', 'color': [255, 0, 0]},
                {'name': 'loaf', 'color': [0, 0, 255]},
                {'name': 'pastry', 'color': [255, 255, 0]},
                {'name': 'defective_bread', 'color': [0, 128, 255]}
            ]
        }

        # Сохраняем конфигурацию по умолчанию
        self.save_cameras_config(default_config)
        return default_config

    def get_zone_config_path(self, camera_id: str) -> Path:
        """Путь к файлу зон для камеры"""
        zones_dir = Path("training_data/zones")
        zones_dir.mkdir(parents=True, exist_ok=True)
        return zones_dir / f"camera_{camera_id}_zones.json"

    def load_zones_for_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Загрузка зон для конкретной камеры"""
        zones_file = self.get_zone_config_path(camera_id)

        if not zones_file.exists():
            return None

        try:
            with open(zones_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Ошибка загрузки зон для камеры {camera_id}: {e}")
            return None

    def save_zones_for_camera(self, camera_id: str, camera_name: str, zones: Dict[str, Any]) -> bool:
        """Сохранение зон для конкретной камеры"""
        zones_file = self.get_zone_config_path(camera_id)

        try:
            zones_config = {
                'camera_id': camera_id,
                'camera_name': camera_name,
                'zones': zones,
                'created': zones_file.stat().st_mtime if zones_file.exists() else None,
                'updated': time.time()
            }

            with open(zones_file, 'w', encoding='utf-8') as f:
                json.dump(zones_config, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Зоны сохранены для камеры {camera_id}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка сохранения зон для камеры {camera_id}: {e}")
            return False