import cv2
import time
from datetime import datetime
from config_loader import ConfigLoader


class CameraTest:
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

        # Пробуем разные RTSP пути
        rtsp_paths = [
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/stream1",
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/stream0",
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/live",
            f"rtsp://{camera_config.login}:{camera_config.password}@{camera_config.camera_ip}/h264",
        ]

        for rtsp_url in rtsp_paths:
            try:
                safe_url = rtsp_url.replace(camera_config.password, '***')
                print(f"   Пробуем: {safe_url}")
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#                cap = cv2.VideoCapture(rtsp_url)
#                cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
#                cap.set(cv2.CAP_PROP_TIMEOUT, 5000)

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
            print("\n🎬 Запускаем предварительный просмотр...")
            self.preview_cameras()

        return success_count > 0

    def preview_cameras(self):
        """Предварительный просмотр с камер"""
        print("👁️  Предварительный просмотр (нажмите 'q' для выхода)")
        print("   [s] - сохранить кадры")
        print("   [c] - проверить подсчет (эмуляция)")

        frame_count = 0

        while True:
            display_frames = []

            for oven_id, conn in self.connections.items():
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

                    display_frames.append(frame)
                    conn['last_frame'] = frame

            # Показываем кадры
            for i, frame in enumerate(display_frames):
                cv2.imshow(f'Camera {list(self.connections.keys())[i]}', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_test_frames()
            elif key == ord('c'):
                self.simulate_counting()

            frame_count += 1
            time.sleep(0.033)

        cv2.destroyAllWindows()

    def save_test_frames(self):
        """Сохранение тестовых кадров"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for oven_id, conn in self.connections.items():
            if 'last_frame' in conn:
                filename = f"test_frame_oven{oven_id}_{timestamp}.jpg"
                cv2.imwrite(filename, conn['last_frame'])
                print(f"💾 Сохранен кадр: {filename}")

    def simulate_counting(self):
        """Эмуляция подсчета"""
        print("🧮 Эмуляция детекции и подсчета...")

        for oven_id, conn in self.connections.items():
            if 'last_frame' in conn:
                frame = conn['last_frame']
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

    def cleanup(self):
        """Очистка ресурсов"""
        for conn in self.connections.values():
            if 'cap' in conn:
                conn['cap'].release()
        cv2.destroyAllWindows()