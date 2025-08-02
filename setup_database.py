from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Enterprise, Workshop, Oven, Product
from config_loader import ConfigLoader


def setup_database_from_config(config_file='cameras.yaml'):
    """Настройка базы данных на основе конфигурации"""

    # Загружаем конфигурацию
    config = ConfigLoader(config_file)
    cameras = config.get_cameras()

    # Создаем БД
    engine = create_engine('sqlite:///bread_production.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    print("🗄️  Настройка базы данных...")

    # Группируем камеры по предприятиям и цехам
    enterprises_data = {}

    for camera in cameras:
        ent_name = camera.enterprise_name
        workshop_name = camera.workshop_name

        if ent_name not in enterprises_data:
            enterprises_data[ent_name] = {}

        if workshop_name not in enterprises_data[ent_name]:
            enterprises_data[ent_name][workshop_name] = []

        enterprises_data[ent_name][workshop_name].append(camera)

    # Создаем структуру в БД
    for ent_name, workshops in enterprises_data.items():
        # Проверяем, есть ли уже предприятие
        enterprise = session.query(Enterprise).filter_by(name=ent_name).first()
        if not enterprise:
            enterprise = Enterprise(
                name=ent_name,
                code=ent_name.replace(" ", "").upper()[:10]
            )
            session.add(enterprise)
            session.flush()  # Получаем ID
            print(f"✅ Создано предприятие: {ent_name}")

        for workshop_name, cameras_in_workshop in workshops.items():
            # Проверяем, есть ли уже цех
            workshop = session.query(Workshop).filter_by(
                name=workshop_name,
                enterprise_id=enterprise.id
            ).first()

            if not workshop:
                workshop = Workshop(
                    name=workshop_name,
                    enterprise_id=enterprise.id
                )
                session.add(workshop)
                session.flush()  # Получаем ID
                print(f"✅ Создан цех: {workshop_name}")

            # Создаем печи
            for camera in cameras_in_workshop:
                # Проверяем, есть ли уже печь
                existing_oven = session.query(Oven).filter_by(
                    number=str(camera.oven_id),
                    workshop_id=workshop.id
                ).first()

                if not existing_oven:
                    oven = Oven(
                        number=str(camera.oven_id),
                        name=camera.oven_name,
                        workshop_id=workshop.id,
                        camera_ip=camera.camera_ip,
                        camera_login=camera.login,
                        camera_password=camera.password
                    )
                    session.add(oven)
                    print(f"✅ Создана печь: {camera.oven_name}")
                else:
                    # Обновляем данные камеры если изменились
                    existing_oven.camera_ip = camera.camera_ip
                    existing_oven.camera_login = camera.login
                    existing_oven.camera_password = camera.password
                    print(f"🔄 Обновлена печь: {camera.oven_name}")

    # Создаем базовые продукты с маркерами
    default_products = [
        {'name': 'Белый хлеб', 'code': 'WB001', 'marker_shape': 'circle'},
        {'name': 'Черный хлеб', 'code': 'BB001', 'marker_shape': 'square'},
        {'name': 'Батон', 'code': 'BT001', 'marker_shape': 'triangle'},
        {'name': 'Булочки', 'code': 'BU001', 'marker_shape': 'diamond'},
        {'name': 'Спецхлеб', 'code': 'SP001', 'marker_shape': 'star'},
    ]

    for prod_data in default_products:
        existing_product = session.query(Product).filter_by(code=prod_data['code']).first()
        if not existing_product:
            product = Product(
                name=prod_data['name'],
                code=prod_data['code'],
                marker_shape=prod_data['marker_shape']
            )
            session.add(product)
            print(f"✅ Создан продукт: {prod_data['name']} ({prod_data['marker_shape']})")

    session.commit()
    session.close()

    print("🎉 База данных настроена на основе cameras.yaml!")
