# install.py - Автоматическая установка с диагностикой
import subprocess
import sys
import platform
import os


def run_command(cmd, description):
    """Выполнение команды с обработкой ошибок"""
    print(f"\n🔄 {description}...")
    print(f"Команда: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                capture_output=True, text=True)
        print(f"✅ {description} - успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ошибка:")
        print(f"Код ошибки: {e.returncode}")
        print(f"Вывод: {e.stdout}")
        print(f"Ошибки: {e.stderr}")
        return False


def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"🐍 Python версия: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 8:
        print("❌ Требуется Python 3.8 или новее")
        return False

    if version.minor > 11:
        print("⚠️  Python 3.12+ может иметь проблемы совместимости")

    return True


def detect_system():
    """Определение операционной системы"""
    system = platform.system().lower()
    print(f"💻 Операционная система: {system}")

    is_windows = system == 'windows'

    if is_windows:
        print("⚠️  Windows обнаружена - Coral TPU недоступен")
        print("   Будет использоваться CPU детекция")

    return is_windows


def install_packages(is_windows=False):
    """Установка пакетов по частям"""
    print("\n📦 УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 50)

    # 1. Обновляем базовые инструменты
    if not run_command(
            f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
            "Обновление pip/setuptools/wheel"
    ):
        return False

    # 2. Базовые пакеты (часто проблемные)
    basic_packages = [
        "numpy>=1.21.0,<1.25.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.0.0"
    ]

    for package in basic_packages:
        if not run_command(
                f"{sys.executable} -m pip install '{package}'",
                f"Установка {package}"
        ):
            print(f"⚠️  Пробуем альтернативную установку {package}...")
            # Пробуем без кеша
            if not run_command(
                    f"{sys.executable} -m pip install --no-cache-dir '{package}'",
                    f"Установка {package} (без кеша)"
            ):
                print(f"❌ Не удалось установить {package}")
                return False

    # 3. Веб и БД пакеты
    web_packages = [
        "SQLAlchemy>=2.0.0",
        "Flask>=2.3.0",
        "Flask-CORS>=4.0.0",
        "requests>=2.28.0",
        "PyYAML>=6.0"
    ]

    for package in web_packages:
        if not run_command(
                f"{sys.executable} -m pip install '{package}'",
                f"Установка {package}"
        ):
            return False

    # 4. Машинное обучение
    ml_packages = [
        "tensorflow-cpu>=2.12.0,<2.16.0" if is_windows else "tensorflow>=2.12.0,<2.16.0",
        "ultralytics>=8.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0"
    ]

    for package in ml_packages:
        if not run_command(
                f"{sys.executable} -m pip install '{package}'",
                f"Установка {package}"
        ):
            print(f"⚠️  Пропускаем {package} - не критично для базовой работы")

    # 5. Coral TPU (только Linux)
    if not is_windows:
        coral_packages = ["pycoral", "tflite-runtime"]
        for package in coral_packages:
            if not run_command(
                    f"{sys.executable} -m pip install '{package}'",
                    f"Установка {package}"
            ):
                print(f"⚠️  {package} не установлен - используйте системный пакет")

    return True


def test_installation():
    """Тестирование установки"""
    print("\n🧪 ТЕСТИРОВАНИЕ УСТАНОВКИ")
    print("=" * 50)

    test_results = {}

    # Список пакетов для тестирования
    tests = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("opencv", "import cv2; print(f'OpenCV {cv2.__version__}')"),
        ("flask", "import flask; print(f'Flask {flask.__version__}')"),
        ("sqlalchemy", "import sqlalchemy; print(f'SQLAlchemy {sqlalchemy.__version__}')"),
        ("yaml", "import yaml; print('PyYAML работает')"),
        ("requests", "import requests; print(f'Requests {requests.__version__}')"),
    ]

    optional_tests = [
        ("tensorflow", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"),
        ("ultralytics", "from ultralytics import YOLO; print('YOLO работает')"),
        ("sklearn", "import sklearn; print(f'Scikit-learn {sklearn.__version__}')"),
        ("matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
    ]

    # Основные тесты
    for name, test_code in tests:
        try:
            exec(test_code)
            test_results[name] = True
            print(f"✅ {name}")
        except Exception as e:
            test_results[name] = False
            print(f"❌ {name}: {e}")

    # Опциональные тесты
    print("\n📊 Опциональные компоненты:")
    for name, test_code in optional_tests:
        try:
            exec(test_code)
            test_results[name] = True
            print(f"✅ {name}")
        except Exception as e:
            test_results[name] = False
            print(f"⚠️  {name}: {e}")

    # Тест создания кадра
    try:
        import numpy as np
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✅ Создание тестового кадра: {frame.shape}")
        test_results['frame_creation'] = True
    except Exception as e:
        print(f"❌ Создание кадра: {e}")
        test_results['frame_creation'] = False

    return test_results


def main():
    """Главная функция установки"""
    print("🥖 АВТОМАТИЧЕСКАЯ УСТАНОВКА СИСТЕМЫ ПОДСЧЕТА ХЛЕБА")
    print("=" * 60)

    # Проверки системы
    if not check_python_version():
        print("❌ Неподходящая версия Python")
        return False

    is_windows = detect_system()

    # Установка пакетов
    if not install_packages(is_windows):
        print("❌ Ошибка установки пакетов")
        return False

    # Тестирование
    test_results = test_installation()

    # Итоговый отчет
    print("\n📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)

    essential_packages = ['numpy', 'opencv', 'flask', 'sqlalchemy', 'frame_creation']
    essential_ok = all(test_results.get(pkg, False) for pkg in essential_packages)

    if essential_ok:
        print("🎉 БАЗОВАЯ УСТАНОВКА УСПЕШНА!")
        print("\n✅ Готово к работе:")
        print("   python quick_test.py      - тест камер")
        if is_windows:
            print("   python main_cpu_test.py   - запуск системы (CPU)")
        else:
            print("   python main_multicamera.py - запуск системы (TPU)")

        # Дополнительные компоненты
        optional_ok = sum(test_results.get(pkg, False)
                          for pkg in ['tensorflow', 'ultralytics', 'sklearn', 'matplotlib'])
        print(f"\n📊 Дополнительных компонентов: {optional_ok}/4")

        if optional_ok < 2:
            print("⚠️  Машинное обучение может не работать")
            print("   Попробуйте: pip install tensorflow-cpu ultralytics")

    else:
        print("❌ УСТАНОВКА НЕ ЗАВЕРШЕНА")
        print("\n❌ Проблемы с базовыми компонентами:")
        for pkg in essential_packages:
            if not test_results.get(pkg, False):
                print(f"   - {pkg}")

        print("\n💡 Попробуйте:")
        print("   1. Обновите Python до 3.9-3.10")
        print("   2. Создайте новое виртуальное окружение")
        print("   3. Используйте conda вместо pip")

    return essential_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)