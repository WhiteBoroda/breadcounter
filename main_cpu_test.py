from config_loader import ConfigLoader


def main():
    print("🥖 СИСТЕМА ПОДСЧЕТА ХЛЕБА - CPU ТЕСТИРОВАНИЕ")
    print("=" * 60)
    print("⚠️  Это упрощенная версия для тестирования без Coral TPU")
    print("🖥️  Используется CPU детекция (медленнее, но работает)")

    try:
        # Загружаем конфигурацию из YAML
        config = ConfigLoader('cameras.yaml')
        cameras = config.get_cameras()

        print(f"\n📹 Найдено {len(cameras)} камер в конфигурации:")
        config.print_config_summary()

        choice = input("Запустить CPU тестирование? (y/n): ")
        if choice.lower() != 'y':
            return

        # Создаем тестовую систему
        system = CPUTestSystem()

        # Добавляем камеры из конфигурации
        connected_cameras = 0
        for camera in cameras:
            if system.add_camera(camera.oven_id, camera.camera_ip, camera.login, camera.password):
                connected_cameras += 1

        if connected_cameras == 0:
            print("❌ Ни одна камера не подключилась")
            return

        print(f"\n✅ Подключено камер: {connected_cameras}/{len(cameras)}")
        print("\n🎬 Запускаем тестирование...")
        print("   Система будет показывать статистику каждые 10 секунд")
        print("   Нажмите Ctrl+C для остановки")

        # Запуск тестирования
        system.start_testing()

    except FileNotFoundError:
        print("❌ Файл cameras.yaml не найден!")
        print("   Создайте файл конфигурации или укажите правильный путь")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
