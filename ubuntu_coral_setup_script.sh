#!/bin/bash
# setup_coral_ubuntu24.sh - Автоматическая настройка Coral TPU на Ubuntu 24.04 с pyenv

set -e  # Прекратить выполнение при ошибке

echo "🥖 Настройка системы подсчета хлеба с Coral TPU"
echo "================================================"
echo "Ubuntu: $(lsb_release -d | cut -f2)"
echo "Пользователь: $USER"
echo ""

# Проверка прав sudo
if ! sudo -v; then
    echo "❌ Требуются права sudo"
    exit 1
fi

# Проверка Coral TPU
echo "🔍 Проверка Coral TPU..."
if lspci | grep -q "Coral Edge TPU"; then
    echo "✅ Coral Edge TPU обнаружен:"
    lspci | grep "Coral Edge TPU"
else
    echo "❌ Coral Edge TPU не найден в системе"
    echo "   Проверьте подключение PCIe карты"
    exit 1
fi

# Обновление системы
echo ""
echo "📦 Обновление системы..."
sudo apt update
sudo apt upgrade -y

# Проверка pyenv
echo ""
echo "🐍 Проверка pyenv..."
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv не найден. Пожалуйста, установите pyenv перед запуском скрипта."
    exit 1
fi

# Установка системных зависимостей
echo ""
echo "🔧 Установка системных библиотек..."

# Базовые инструменты разработки
sudo apt install -y build-essential cmake git pkg-config

# OpenCV зависимости
sudo apt install -y libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev libwebp-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev libxine2-dev
sudo apt install -y libatlas-base-dev libeigen3-dev libtbb-dev
sudo apt install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt install -y libvorbis-dev libxvidcore-dev libopencore-amrnb-dev
sudo apt install -y libopencore-amrwb-dev libswresample-dev
# Добавляем зависимости для _ctypes и других модулей
sudo apt install -y libffi-dev libbz2-dev liblzma-dev libsqlite3-dev libreadline-dev libssl-dev

# Дополнительные утилиты
sudo apt install -y htop tree curl wget unzip

# Установка репозитория Google Coral
echo ""
echo "🧠 Настройка репозитория Google Coral..."

# Добавление ключа и репозитория
if [ ! -f /etc/apt/sources.list.d/coral-edgetpu.list ]; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update
    echo "✅ Репозиторий Coral добавлен"
else
    echo "ℹ️  Репозиторий Coral уже настроен"
fi

# Установка Coral TPU runtime
echo ""
echo "🚀 Установка Coral TPU runtime..."

# Устанавливаем стандартную частоту (рекомендуется)
sudo apt install -y libedgetpu1-std

# Edge TPU Compiler
sudo apt install -y edgetpu-compiler

# Драйверы для PCIe устройств
echo ""
echo "🔌 Настройка драйверов PCIe..."

# Gasket драйвер для PCIe Coral
sudo apt install -y gasket-dkms linux-headers-$(uname -r)

# Добавление пользователя в группу apex
sudo usermod -a -G apex $USER

# Настройка udev правил
echo 'SUBSYSTEM=="apex", MODE="0660", GROUP="apex"' | sudo tee /etc/udev/rules.d/65-apex.rules
sudo udevadm control --reload-rules

# Создание рабочей директории
echo ""
echo "📁 Создание рабочей директории..."

WORK_DIR="$HOME/breadcounter"
if [ ! -d "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    echo "✅ Создана директория: $WORK_DIR"
else
    cd "$WORK_DIR"
    echo "ℹ️  Используется существующая директория: $WORK_DIR"
fi

# Настройка pyenv local
echo ""
echo "🐍 Настройка pyenv local на 3.9.18"
pyenv local 3.9.18

# Проверка, что pyenv использует нужную версию
echo "✅ Проверка текущей версии Python: $(python --version)"
if [[ "$(python --version)" != "Python 3.9.18" ]]; then
    echo "❌ pyenv не использует Python 3.9.18. Убедитесь, что эта версия установлена через pyenv и что pyenv настроен правильно."
    exit 1
fi

# Создание виртуального окружения Python 3.9
echo ""
echo "🐍 Создание виртуального окружения..."

if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Виртуальное окружение создано"
else
    echo "ℹ️  Виртуальное окружение уже существует"
fi

# Активация окружения и установка пакетов
source venv/bin/activate

# Обновление pip
echo ""
echo "📦 Обновление pip и базовых инструментов..."
pip install --upgrade pip setuptools wheel

# Создание requirements.txt если не существует
if [ ! -f "requirements.txt" ]; then
    echo "📝 Создание requirements.txt..."
    cat > requirements.txt << 'EOF'
# Базовые ML пакеты
tensorflow>=2.16.0,<2.17.0
numpy>=1.24.0,<1.27.0
opencv-python>=4.9.0,<4.11.0
Pillow>=10.2.0,<11.0.0

# Веб фреймворк
Flask>=3.0.0,<3.1.0
Flask-CORS>=4.0.0,<5.0.0
gunicorn>=21.2.0,<22.0.0

# База данных
SQLAlchemy>=2.0.25,<2.1.0

# Конфигурация
PyYAML>=6.0.1,<7.0.0
requests>=2.31.0,<3.0.0

# YOLO и детекция
ultralytics>=8.1.0,<8.2.0
scikit-learn>=1.4.0,<1.5.0

# Визуализация
matplotlib>=3.8.0,<3.9.0
seaborn>=0.13.0,<0.14.0

# Утилиты
colorlog>=6.8.0,<7.0.0
psutil>=5.9.0,<6.0.0

# Coral TPU Python bindings
pycoral>=2.0.0,<3.0.0
tflite-runtime>=2.16.0,<2.17.0
EOF
fi

# Установка Python пакетов
echo ""
echo "📦 Установка Python пакетов..."
pip install -r requirements.txt

# Создание тестового скрипта
echo ""
echo "🧪 Создание тестового скрипта..."

cat > test_coral.py << 'EOF'
#!/usr/bin/env python3
# test_coral.py - Тест Coral TPU

print("🧪 Тестирование установки Coral TPU...")

# Проверка базовых библиотек
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: {e}")

# Проверка Coral TPU
try:
    from pycoral.utils import edgetpu
    from pycoral.utils import dataset

    print(f"✅ PyCoral импортирован")

    # Поиск TPU устройств
    devices = edgetpu.list_edge_tpus()
    print(f"🧠 Найдено TPU устройств: {len(devices)}")

    for i, device in enumerate(devices):
        print(f"   Устройство {i}: {device}")

    if devices:
        print("🎉 Coral TPU готов к работе!")
    else:
        print("⚠️  TPU устройства не найдены")
        print("   Возможно требуется перезагрузка системы")

except ImportError as e:
    print(f"❌ PyCoral не доступен: {e}")
    print("   Проверьте установку: sudo apt install python3-pycoral")

# Тест создания случайного кадра
try:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"✅ Тестовый кадр создан: {frame.shape}")
except Exception as e:
    print(f"❌ Ошибка создания кадра: {e}")

print("\n🎯 Тестирование завершено!")
EOF

chmod +x test_coral.py

# Запуск теста
echo ""
echo "🚀 Запуск тестирования..."
python test_coral.py

# Итоговые инструкции
echo ""
echo "✅ УСТАНОВКА ЗАВЕРШЕНА!"
echo "======================="
echo ""
echo "📍 Рабочая директория: $WORK_DIR"
echo "🐍 Python окружение: $WORK_DIR/venv"
echo ""
echo "🔄 ВАЖНО: Перезагрузите систему для активации драйверов:"
echo "   sudo reboot"
echo ""
echo "🚀 После перезагрузки запустите:"
echo "   cd $WORK_DIR"
echo "   source venv/bin/activate"
echo "   python test_coral.py"
echo ""
echo "📁 Скопируйте файлы системы в: $WORK_DIR"
echo "   - config_loader.py"
echo "   - models.py"
echo "   - coral_detector.py"
echo "   - main_multicamera.py"
echo "   - cameras.yaml"
echo "   - и остальные файлы системы"
echo ""
echo "🎯 Затем запустите тестирование камер:"
echo "   python quick_test.py"
echo ""

# Предупреждение о перезагрузке
echo "⚠️  ПЕРЕЗАГРУЗКА ОБЯЗАТЕЛЬНА для активации драйверов Coral TPU!"
echo "   Выполните: sudo reboot"