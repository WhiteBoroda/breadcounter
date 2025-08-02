import cv2
import time
from datetime import datetime
from config_loader import ConfigLoader


class CameraTestHeadless:
    def __init__(self, config_file='cameras.yaml'):
        # Загружаем конфигурацию из YAML
        self.config = ConfigLoader(config_file)
        self.cameras = {cam.oven_id: cam for cam in self.config.get_cameras()}
        self.connections = {}

        print("📋 Загружена конфигурация камер:")
        self.config.print_config_summary()

    def test_camera_connection(self, oven_id, camera_config):
        """Тестирование подключения к одной камере"""
        print(f"🔌 Тестируем подключение к {camera_config.oven_name} ({camera_config.camera_ip})...")

        # Пробуем разные RTSP пути согласно инструкции камеры
        rtsp_paths = [
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}:554/ch01/0",  # основной поток
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}:554/ch01/1",  # дополнительный поток
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}:554/ch01/2",  # мобильный поток
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/cam/realmonitor?channel=1&subtype=0",
        ]

        for rtsp_url in rtsp_paths:
            try:
                safe_url = rtsp_url.replace(camera_config.password, '***')
                print(f"   Пробуем: {safe_url}")
                cap = cv2.VideoCapture(rtsp_url)
                # cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)  # Не поддерживается в этой версии OpenCV

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"   ✅ {camera_config.oven_name}: Подключение успешно!")
                        print(f"      Разрешение: {frame.shape[1]}x{frame.shape[0]}")

                        self.connections[oven_id] = {
                            'cap': cap,
                            'url': rtsp_url,
                            'config': camera_config,
                            'last_frame': frame
                        }
                        return True
                    else:
                        cap.release()
                else:
                    cap.release()

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                continue

        print(f"   ❌ {camera_config.oven_name}: Не удалось подключиться")
        return False

    def test_all_cameras(self):
        """Тестирование всех камер из конфигурации"""
        print("🚀 ТЕСТИРОВАНИЕ ПОДКЛЮЧЕНИЯ К КАМЕРАМ")
        print("=" * 50)

        success_count = 0

        for oven_id, camera_config in self.cameras.items():
            if self.test_camera_connection(oven_id, camera_config):
                success_count += 1
            print()

        print(f"📊 Результат: {success_count}/{len(self.cameras)} камер подключено")

        if success_count > 0:
            print("\n📸 Сохраняем тестовые кадры...")
            self.save_test_frames()
            
            print("\n🧮 Эмулируем детекцию...")
            self.simulate_counting()

        return success_count > 0

    def save_test_frames(self):
        """Сохранение тестовых кадров"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for oven_id, conn in self.connections.items():
            # Получаем свежий кадр
            ret, frame = conn['cap'].read()
            if ret:
                config = conn['config']
                
                # Добавляем информацию на кадр
                cv2.putText(frame, f"{config.oven_name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"IP: {config.camera_ip}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Цех: {config.workshop_name}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Рисуем тестовые зоны подсчета
                h, w = frame.shape[:2]
                line_positions = [int(h * 0.3), int(h * 0.5), int(h * 0.7)]

                for i, y in enumerate(line_positions):
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
                    cv2.line(frame, (0, y), (w, y), color, 2)
                    cv2.putText(frame, f"Zone {i + 1}", (w - 100, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                filename = f"test_frame_oven{oven_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Сохранен кадр: {filename}")

    def simulate_counting(self):
        """Эмуляция подсчета"""
        print("🧮 Эмуляция детекции и подсчета...")

        for oven_id, conn in self.connections.items():
            ret, frame = conn['cap'].read()
            if ret:
                config = conn['config']

                # Простая эмуляция подсчета
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bread_objects = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 5000 < area < 50000:
                        bread_objects += 1

                print(f"   {config.oven_name}: Обнаружено ~{bread_objects} объектов")

    def test_stream_quality(self, duration_seconds=10):
        """Тест качества потока"""
        print(f"\n📡 Тест качества потока ({duration_seconds} секунд)...")
        
        for oven_id, conn in self.connections.items():
            config = conn['config']
            cap = conn['cap']
            
            print(f"\n🔥 {config.oven_name}:")
            
            frame_count = 0
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            while time.time() < end_time:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                time.sleep(0.1)  # 10 FPS тест
            
            actual_duration = time.time() - start_time
            fps = frame_count / actual_duration
            
            print(f"   📊 Кадров получено: {frame_count}")
            print(f"   🎬 FPS: {fps:.1f}")
            print(f"   ⏱️  Время: {actual_duration:.1f} сек")

    def cleanup(self):
        """Очистка ресурсов"""
        for conn in self.connections.values():
            if 'cap' in conn:
                conn['cap'].release()


def main():
    """Главная функция"""
    try:
        tester = CameraTestHeadless()
        
        if tester.test_all_cameras():
            print("\n✅ Камеры работают! Можно переходить к следующему этапу:")
            print("   1. Сбор данных: python training_pipeline.py")
            print("   2. CPU система: python main_cpu_test.py")
            
            # Дополнительные тесты
            choice = input("\nЗапустить тест качества потока? (y/n): ").strip().lower()
            if choice == 'y':
                tester.test_stream_quality()
        else:
            print("\n❌ Проблемы с подключением к камерам")
            
    except FileNotFoundError:
        print("❌ Файл cameras.yaml не найден!")
        print("   Создайте файл конфигурации")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if 'tester' in locals():
            tester.cleanup()


if __name__ == "__main__":
    main()
