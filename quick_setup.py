# quick_setup.py - Быстрый запуск всех компонентов системы
import subprocess
import sys
import time
import threading
import os
from pathlib import Path


class QuickSetup:
    """Быстрый запуск системы подсчета хлеба"""

    def __init__(self):
        self.processes = {}
        self.running = True

    def print_banner(self):
        """Вывод баннера"""
        print("🥖" + "=" * 58 + "🥖")
        print("🥖            СИСТЕМА ПОДСЧЕТА ХЛЕБА v2.0              🥖")
        print("🥖                Быстрый запуск                      🥖")
        print("🥖" + "=" * 58 + "🥖")
        print()

    def check_requirements(self):
        """Проверка требований"""
        print("🔍 Проверка системных требований...")

        # Проверяем Python
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            print(f"❌ Требуется Python 3.8+, найден {python_version.major}.{python_version.minor}")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Проверяем ключевые файлы
        required_files = [
            'cameras.yaml',
            'main_multicamera.py',
#            'quick_test_headless.py',
            'improved_interactive_training_web.py',
            'web_api.py'
        ]

        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ Отсутствует файл: {file}")
                return False
        print("✅ Все ключевые файлы найдены")

        # Проверяем папки
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('training_data/images', exist_ok=True)
        os.makedirs('training_data/annotations', exist_ok=True)
        print("✅ Структура папок готова")

        return True

    def show_menu(self):
        """Главное меню"""
        while self.running:
            print("\n🎯 МЕНЮ БЫСТРОГО ЗАПУСКА")
            print("=" * 40)
            print("1. 🧪 Тест системы (TPU + камеры)")
            print("2. 🧠 Веб-интерфейс обучения (порт 5001)")
            print("3. 📊 Веб-мониторинг (порт 5000)")
            print("4. 🚀 Полная система (многокамерная)")
            print("5. 🎬 Обработка больших видео")
            print("6. 📈 Все веб-интерфейсы")
            print("7. 🔧 Диагностика")
            print("0. ❌ Выход")
            print("=" * 40)

            try:
                choice = input("Выберите опцию (0-7): ").strip()

                if choice == '0':
                    self.shutdown()
                    break
                elif choice == '1':
                    self.run_system_test()
                elif choice == '2':
                    self.run_training_interface()
                elif choice == '3':
                    self.run_monitoring()
                elif choice == '4':
                    self.run_full_system()
                elif choice == '5':
                    self.run_video_processor()
                elif choice == '6':
                    self.run_all_web_interfaces()
                elif choice == '7':
                    self.run_diagnostics()
                else:
                    print("❌ Неверная опция")

            except KeyboardInterrupt:
                print("\n🛑 Завершение работы...")
                self.shutdown()
                break

    def run_system_test(self):
        """Тест системы"""
        print("\n🧪 ТЕСТ СИСТЕМЫ")
        print("-" * 30)

        # Тест TPU
        print("1. Тест Coral TPU...")
        try:
            result = subprocess.run([sys.executable, 'test_tpu.py'],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ TPU тест пройден")
            else:
                print("⚠️  TPU тест с предупреждениями")
                print(result.stdout[-200:])  # Последние 200 символов
        except Exception as e:
            print(f"❌ Ошибка TPU теста: {e}")

        # Тест камер
        print("\n2. Тест камер...")
        try:
            result = subprocess.run([sys.executable, 'quick_test_headless.py'],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("✅ Камеры протестированы")
                # Выводим краткий результат
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Результат:' in line or 'подключено' in line:
                        print(f"   {line}")
            else:
                print("⚠️  Проблемы с камерами")
        except Exception as e:
            print(f"❌ Ошибка теста камер: {e}")

        print("\n✅ Тестирование завершено")
        input("Нажмите Enter для продолжения...")

    def run_training_interface(self):
        """Запуск веб-интерфейса обучения"""
        print("\n🧠 ЗАПУСК ВЕБ-ИНТЕРФЕЙСА ОБУЧЕНИЯ")
        print("-" * 40)
        print("🌐 URL: http://localhost:5001")
        print("📁 Поддержка файлов до 5GB")
        print("⚡ Чанковая загрузка")

        if self._confirm_start():
            self._start_process('training_web',
                                [sys.executable, 'improved_interactive_training_web.py'])

    def run_monitoring(self):
        """Запуск веб-мониторинга"""
        print("\n📊 ЗАПУСК ВЕБ-МОНИТОРИНГА")
        print("-" * 30)
        print("🌐 URL: http://localhost:5000")
        print("📈 Мониторинг в реальном времени")

        if self._confirm_start():
            self._start_process('monitoring',
                                [sys.executable, 'web_api.py'])

    def run_full_system(self):
        """Запуск полной многокамерной системы"""
        print("\n🚀 ЗАПУСК ПОЛНОЙ СИСТЕМЫ")
        print("-" * 30)
        print("🧠 Coral TPU обработка")
        print("📹 Многокамерный мониторинг")
        print("🥖 Умное определение партий")
        print("📊 Веб-интерфейс мониторинга")

        # Проверяем модель
        if not os.path.exists('bread_detector_edgetpu.tflite'):
            print("\n⚠️  ВНИМАНИЕ: Модель не найдена!")
            print("   Сначала обучите модель через веб-интерфейс")
            print("   или скопируйте готовую модель")

            if not self._confirm_start():
                return

        if self._confirm_start():
            # Запускаем основную систему
            self._start_process('main_system',
                                [sys.executable, 'main_multicamera.py', 'cameras.yaml'])

            # Через 5 секунд запускаем веб-мониторинг
            time.sleep(5)
            if 'main_system' in self.processes:
                print("🌐 Запуск веб-мониторинга...")
                self._start_process('monitoring',
                                    [sys.executable, 'web_api.py'])

    def run_video_processor(self):
        """Запуск обработчика больших видео"""
        print("\n🎬 ОБРАБОТКА БОЛЬШИХ ВИДЕО")
        print("-" * 30)

        # Запрашиваем параметры
        video_path = input("Путь к видео файлу (или папке): ").strip()
        if not video_path:
            print("❌ Путь не указан")
            return

        max_frames = input("Максимум кадров (по умолчанию 1000): ").strip()
        max_frames = max_frames if max_frames.isdigit() else "1000"

        strategy = input("Стратегия (uniform/smart/quality_based, по умолчанию smart): ").strip()
        strategy = strategy if strategy in ['uniform', 'smart', 'quality_based'] else 'smart'

        # Формируем команду
        if os.path.isfile(video_path):
            cmd = [sys.executable, 'large_video_processor.py',
                   '--video', video_path, '--max-frames', max_frames, '--strategy', strategy]
        elif os.path.isdir(video_path):
            cmd = [sys.executable, 'large_video_processor.py',
                   '--videos-dir', video_path, '--max-frames', max_frames, '--strategy', strategy]
        else:
            print("❌ Указанный путь не существует")
            return

        print(f"\n🚀 Запуск обработки...")
        print(f"   Команда: {' '.join(cmd[2:])}")

        if self._confirm_start():
            self._start_process('video_processor', cmd)

    def run_all_web_interfaces(self):
        """Запуск всех веб-интерфейсов"""
        print("\n🌐 ЗАПУСК ВСЕХ ВЕБ-ИНТЕРФЕЙСОВ")
        print("-" * 40)
        print("🧠 Обучение: http://localhost:5001")
        print("📊 Мониторинг: http://localhost:5000")

        if self._confirm_start():
            # Запускаем интерфейс обучения
            self._start_process('training_web',
                                [sys.executable, 'improved_interactive_training_web.py'])

            time.sleep(2)

            # Запускаем мониторинг
            self._start_process('monitoring',
                                [sys.executable, 'web_api.py'])

            print("\n✅ Все веб-интерфейсы запущены")

    def run_diagnostics(self):
        """Запуск диагностики"""
        print("\n🔧 ДИАГНОСТИКА СИСТЕМЫ")
        print("-" * 30)

        # Версии библиотек
        print("📋 Версии библиотек:")
        try:
            import cv2
            print(f"   OpenCV: {cv2.__version__}")
        except:
            print("   OpenCV: ❌ Не установлен")

        try:
            import numpy as np
            print(f"   NumPy: {np.__version__}")
        except:
            print("   NumPy: ❌ Не установлен")

        try:
            import flask
            print(f"   Flask: {flask.__version__}")
        except:
            print("   Flask: ❌ Не установлен")

        # Проверка TPU
        print("\n🧠 Coral TPU:")
        try:
            from pycoral.utils import edgetpu
            devices = edgetpu.list_edge_tpus()
            print(f"   Устройств найдено: {len(devices)}")
            for i, device in enumerate(devices):
                print(f"   Устройство {i}: {device}")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

        # Проверка дискового пространства
        print("\n💾 Дисковое пространство:")
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            print(f"   Всего: {total // (1024 ** 3)} GB")
            print(f"   Свободно: {free // (1024 ** 3)} GB")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

        input("Нажмите Enter для продолжения...")

    def _confirm_start(self):
        """Подтверждение запуска"""
        choice = input("\nЗапустить? (y/n): ").strip().lower()
        return choice in ['y', 'yes', 'да', 'д']

    def _start_process(self, name, cmd):
        """Запуск процесса"""
        try:
            print(f"🚀 Запуск {name}...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            self.processes[name] = process

            # Запускаем мониторинг вывода в отдельном потоке
            threading.Thread(target=self._monitor_process,
                             args=(name, process), daemon=True).start()

            print(f"✅ {name} запущен (PID: {process.pid})")

            # Показываем меню управления процессами
            self._show_process_menu()

        except Exception as e:
            print(f"❌ Ошибка запуска {name}: {e}")

    def _monitor_process(self, name, process):
        """Мониторинг вывода процесса"""
        try:
            for line in process.stdout:
                print(f"[{name}] {line.strip()}")
        except:
            pass

    def _show_process_menu(self):
        """Меню управления процессами"""
        while self.processes:
            print(f"\n🔧 УПРАВЛЕНИЕ ПРОЦЕССАМИ ({len(self.processes)} активных)")
            print("-" * 40)

            for i, (name, process) in enumerate(self.processes.items(), 1):
                status = "🟢 Работает" if process.poll() is None else "🔴 Остановлен"
                print(f"{i}. {name}: {status} (PID: {process.pid})")

            print(f"{len(self.processes) + 1}. 🛑 Остановить все")
            print(f"{len(self.processes) + 2}. ⬅️  Назад в главное меню")

            try:
                choice = input("Выберите действие: ").strip()

                if choice.isdigit():
                    choice_num = int(choice)
                    process_names = list(self.processes.keys())

                    if 1 <= choice_num <= len(process_names):
                        # Остановить конкретный процесс
                        name = process_names[choice_num - 1]
                        self._stop_process(name)
                    elif choice_num == len(process_names) + 1:
                        # Остановить все
                        self._stop_all_processes()
                        break
                    elif choice_num == len(process_names) + 2:
                        # Назад в меню
                        break

            except (ValueError, KeyboardInterrupt):
                break

    def _stop_process(self, name):
        """Остановка конкретного процесса"""
        if name in self.processes:
            process = self.processes[name]
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} остановлен")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔥 {name} принудительно завершен")
            except Exception as e:
                print(f"❌ Ошибка остановки {name}: {e}")
            finally:
                del self.processes[name]

    def _stop_all_processes(self):
        """Остановка всех процессов"""
        print("🛑 Остановка всех процессов...")
        for name in list(self.processes.keys()):
            self._stop_process(name)
        print("✅ Все процессы остановлены")

    def shutdown(self):
        """Завершение работы"""
        self.running = False
        self._stop_all_processes()
        print("\n🎉 Система завершена. До свидания!")


def main():
    """Главная функция"""
    setup = QuickSetup()

    try:
        setup.print_banner()

        if not setup.check_requirements():
            print("❌ Системные требования не выполнены")
            return

        setup.show_menu()

    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал прерывания")
        setup.shutdown()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        setup.shutdown()


if __name__ == "__main__":
    main()