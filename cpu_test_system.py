class CPUTestSystem:
    """Упрощенная система для тестирования без TPU"""

    def __init__(self):
        self.cameras = {}
        self.running = False

    def add_camera(self, oven_id, camera_ip, login, password):
        """Добавление камеры"""
        processor = CPUCameraProcessor(camera_ip, login, password, oven_id)

        if processor.connect_camera():
            self.cameras[oven_id] = processor
            print(f"✅ Камера печи {oven_id} добавлена")
            return True
        else:
            print(f"❌ Не удалось добавить камеру печи {oven_id}")
            return False

    def start_testing(self):
        """Запуск тестирования"""
        print("🚀 Запуск CPU тестирования...")

        for processor in self.cameras.values():
            processor.start_processing()

        self.running = True

        try:
            self._monitoring_loop()
        except KeyboardInterrupt:
            print("\n🛑 Остановка тестирования...")
            self.stop()

    def _monitoring_loop(self):
        """Цикл мониторинга"""
        while self.running:
            print("\n" + "=" * 60)
            print("📊 СТАТИСТИКА CPU ТЕСТИРОВАНИЯ")
            print("=" * 60)

            for oven_id, processor in self.cameras.items():
                results = processor.get_latest_results()

                if results:
                    detections = results['detections']
                    performance = results['performance']

                    bread_count = len([d for d in detections if d['class_name'] == 'bread'])
                    marker_count = len([d for d in detections if d['class_name'] in ['circle', 'square', 'triangle']])

                    print(f"\n🔥 Печь {oven_id}:")
                    print(f"   📹 FPS: {processor.current_fps:2d}")
                    print(f"   🧠 Обработка: {performance['fps']:.1f} FPS")
                    print(f"   🥖 Хлеб: {bread_count} шт")
                    print(f"   🎯 Маркеры: {marker_count} шт")
                    print(f"   ⏱️  Время детекции: {performance['avg_inference_time'] * 1000:.1f}мс")
                else:
                    print(f"\n🔥 Печь {oven_id}: Нет данных")

            time.sleep(10)  # Обновление каждые 10 секунд

    def stop(self):
        """Остановка системы"""
        self.running = False
        for processor in self.cameras.values():
            processor.stop_processing()
        print("✅ CPU тестирование остановлено")