# models.py - Модели базы данных
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Enterprise(Base):
    __tablename__ = 'enterprises'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(10), unique=True)

    workshops = relationship("Workshop", back_populates="enterprise")


class Workshop(Base):
    __tablename__ = 'workshops'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    enterprise_id = Column(Integer, ForeignKey('enterprises.id'))

    enterprise = relationship("Enterprise", back_populates="workshops")
    ovens = relationship("Oven", back_populates="workshop")


class Oven(Base):
    __tablename__ = 'ovens'

    id = Column(Integer, primary_key=True)
    number = Column(String(10), nullable=False)
    name = Column(String(100))
    workshop_id = Column(Integer, ForeignKey('workshops.id'))
    camera_ip = Column(String(15))
    camera_login = Column(String(50))
    camera_password = Column(String(50))
    calibration_data = Column(Text)  # JSON с калибровочными данными

    workshop = relationship("Workshop", back_populates="ovens")
    shift_tasks = relationship("ShiftTask", back_populates="oven")
    production_batches = relationship("ProductionBatch", back_populates="oven")


class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), unique=True)
    marker_shape = Column(String(20))  # circle, square, triangle, etc.
    min_size = Column(Integer, default=8000)  # минимальная площадь в пикселях
    max_size = Column(Integer, default=25000)  # максимальная площадь
    color_profile = Column(Text)  # JSON с цветовыми характеристиками


class ShiftTask(Base):
    __tablename__ = 'shift_tasks'

    id = Column(Integer, primary_key=True)
    oven_id = Column(Integer, ForeignKey('ovens.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    planned_quantity = Column(Integer)
    sequence_order = Column(Integer)
    shift_date = Column(DateTime)
    completed = Column(Boolean, default=False)

    oven = relationship("Oven", back_populates="shift_tasks")
    product = relationship("Product")


class ProductionBatch(Base):
    __tablename__ = 'production_batches'

    id = Column(Integer, primary_key=True)
    oven_id = Column(Integer, ForeignKey('ovens.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime)
    total_count = Column(Integer, default=0)
    defect_count = Column(Integer, default=0)
    tracking_confidence = Column(Float, default=0.0)  # средняя уверенность трекинга

    oven = relationship("Oven", back_populates="production_batches")
    product = relationship("Product")


class DetectionEvent(Base):
    __tablename__ = 'detection_events'

    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey('production_batches.id'))
    timestamp = Column(DateTime, default=datetime.now)
    track_id = Column(Integer)  # ID трека объекта
    event_type = Column(String(20))  # 'bread_detected', 'bread_counted', 'defect_detected'
    confidence = Column(Float)
    bbox_data = Column(Text)  # JSON с координатами
    is_defective = Column(Boolean, default=False)