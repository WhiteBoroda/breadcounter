# main.py
"""Главный файл запуска системы подсчета хлеба"""

import sys
import os
import threading
import time
from multiprocessing import Process

# Добавляем пути к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_main_app():
    """Запуск основного веб-интерфейса"""
    from web.main_app import ProductionMonitorApp
    app = ProductionMonitorApp()
    print("🚀 Запуск основного интерфейса на http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)


def run_training_module():
    """Запуск обучающего модуля"""
    from web.training_module_advanced import AdvancedTrainingModule
    module = AdvancedTrainingModule()
    print("🧠 Запуск обучающего модуля на http://localhost:5001")
    module.run(host='0.0.0.0', port=5001, debug=False)


def check_requirements():
    """Проверка наличия необходимых компонентов"""
    try:
        from core.imports import check_critical_imports
        check_critical_imports()
        print("✅ Критические импорты проверены")

        from core.tpu_manager import TPUManager
        tpu_manager = TPUManager()
        if tpu_manager.is_available():
            device_count = tpu_manager.get_device_count()
            print(f"✅ Coral TPU доступен: {device_count} устройств")
        else:
            print("⚠️  Coral TPU недоступен, будет использован CPU")

        return True

    except Exception as e:
        print(f"❌ Ошибка проверки требований: {e}")
        return False


def create_directories():
    """Создание необходимых директорий"""
    directories = [
        'uploads',
        'temp_uploads',
        'training_data',
        'training_data/images',
        'training_data/annotations',
        'config'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("📁 Директории созданы")


def create_default_config():
    """Создание конфигурации по умолчанию"""
    cameras_config = """cameras:
  - oven_id: 1
    camera_ip: "192.168.1.100"
    login: "admin"
    password: "CHANGE_ME"
    oven_name: "Печь №1"
    workshop_name: "Цех №1"
    enterprise_name: "Хлебозавод"
    product_type: "bread"

system:
  tpu_devices: 1
  frame_rate: 15
  detection_threshold: 0.5
  tracking_max_distance: 100

data_collection:
  output_dir: "training_data"
  save_interval: 5
  video_duration: 30

classes:
  - name: "bread"
    color: [0, 255, 0]
  - name: "bun" 
    color: [255, 0, 0]
  - name: "loaf"
    color: [0, 0, 255]
  - name: "pastry"
    color: [255, 255, 0]
  - name: "defective_bread"
    color: [0, 128, 255]
"""

    config_path = 'config/cameras.yaml'
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(cameras_config)
        print("📝 Создана конфигурация cameras.yaml")
        print("⚠️  ВАЖНО: Измените пароли камер в config/cameras.yaml!")


def main():
    """Главная функция запуска"""
    print("🥖 СИСТЕМА ПОДСЧЕТА ХЛЕБА")
    print("=" * 50)

    # Проверка требований
    if not check_requirements():
        print("❌ Не удалось запустить систему")
        sys.exit(1)

    # Создание директорий и конфигурации
    create_directories()
    create_default_config()

    print("\n🔧 Подготовка к запуску...")
    time.sleep(1)

    try:
        # Запуск процессов
        print("\n🚀 Запуск веб-интерфейсов...")

        # Процесс основного интерфейса
        main_process = Process(target=run_main_app)
        main_process.start()

        # Небольшая задержка перед запуском второго процесса
        time.sleep(2)

        # Процесс обучающего модуля
        training_process = Process(target=run_training_module)
        training_process.start()

        print("\n✅ Система запущена!")
        print("📋 Доступные интерфейсы:")
        print("   🏠 Главная панель: http://localhost:5000")
        print("   🧠 Обучение модели: http://localhost:5001/training")
        print("\n⌨️  Нажмите Ctrl+C для остановки")

        # Ожидание завершения процессов
        try:
            main_process.join()
            training_process.join()
        except KeyboardInterrupt:
            print("\n🛑 Остановка системы...")
            main_process.terminate()
            training_process.terminate()
            main_process.join()
            training_process.join()
            print("✅ Система остановлена")

        return 0

    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено пользователем.")
        exit_code = main()
        sys.exit(exit_code)