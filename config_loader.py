# config_loader.py - Единый загрузчик конфигурации
import yaml
import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class CameraConfig:
    oven_id: int
    camera_ip: str
    login: str
    password: str
    oven_name: str
    workshop_name: str
    enterprise_name: str


class ConfigLoader:
    """Единый загрузчик конфигурации из YAML"""

    def __init__(self, config_file='cameras.yaml'):
        self.config_file = config_file
        self._config_data = None
        self.load_config()

    def load_config(self):
        """Загрузка конфигурации из файла"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Конфигурационный файл {self.config_file} не найден")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            self._config_data = yaml.safe_load(f)

        print(f"✅ Конфигурация загружена из {self.config_file}")

    def get_cameras(self) -> List[CameraConfig]:
        """Получение списка камер"""
        cameras = []

        for cam_data in self._config_data.get('cameras', []):
            camera = CameraConfig(
                oven_id=cam_data['oven_id'],
                camera_ip=cam_data['camera_ip'],
                login=cam_data['login'],
                password=cam_data['password'],
                oven_name=cam_data['oven_name'],
                workshop_name=cam_data['workshop_name'],
                enterprise_name=cam_data['enterprise_name']
            )
            cameras.append(camera)

        return cameras

    def get_system_settings(self) -> Dict[str, Any]:
        """Получение системных настроек"""
        return self._config_data.get('system', {
            'tpu_devices': 1,
            'frame_rate': 15,
            'detection_threshold': 0.5,
            'tracking_max_distance': 100
        })

    def get_classes(self) -> List[Dict[str, Any]]:
        """Получение классов для детекции"""
        return self._config_data.get('classes', [
            {'name': 'bread', 'color': [0, 255, 0]},
            {'name': 'circle', 'color': [255, 0, 0]},
            {'name': 'square', 'color': [0, 0, 255]},
            {'name': 'triangle', 'color': [255, 255, 0]},
            {'name': 'defective_bread', 'color': [0, 128, 255]}
        ])

    def get_camera_by_oven_id(self, oven_id: int) -> CameraConfig:
        """Получение камеры по ID печи"""
        cameras = self.get_cameras()
        for camera in cameras:
            if camera.oven_id == oven_id:
                return camera
        raise ValueError(f"Камера с oven_id={oven_id} не найдена")

    def print_config_summary(self):
        """Вывод сводки конфигурации"""
        cameras = self.get_cameras()
        print("\n📋 СВОДКА КОНФИГУРАЦИИ")
        print("=" * 50)

        for camera in cameras:
            print(f"🔥 {camera.oven_name}")
            print(f"   📹 IP: {camera.camera_ip}")
            print(f"   🏭 Цех: {camera.workshop_name}")
            print(f"   🏢 Предприятие: {camera.enterprise_name}")
            print()
