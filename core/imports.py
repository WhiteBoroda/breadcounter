# core/imports.py
"""Централизованные импорты для проекта подсчета хлеба"""

# Стандартные библиотеки
import os
import sys
import time
import json
import threading
import signal
import logging
import uuid
import base64
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import concurrent.futures
import queue

# Базовые ML библиотеки
import cv2
import numpy as np

# Опциональные импорты с проверкой доступности
try:
    from pycoral.utils import edgetpu, dataset
    from pycoral.adapters import common, detect
    import tflite_runtime.interpreter as tflite

    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from flask import Flask, render_template, render_template_string, request, jsonify, send_file, url_for
    from flask_cors import CORS
    from werkzeug.utils import secure_filename

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Проверка критических зависимостей
def check_critical_imports():
    """Проверка наличия критических библиотек"""
    missing = []

    if 'cv2' not in globals():
        missing.append('opencv-python')
    if 'np' not in globals():
        missing.append('numpy')

    if missing:
        raise ImportError(f"Критические библиотеки не найдены: {missing}")

    return True