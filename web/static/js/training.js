// web/static/js/training.js
// Основной JavaScript для модуля обучения

// Константы
const CHUNK_SIZE = 1024 * 1024; // 1MB чанки

// Глобальные переменные
let isUploading = false;
let uploadSessionId = null;
let currentVideo = null;
let currentCamera = {id: 'default', name: 'Неизвестная камера'};
let totalFrames = 0;
let currentFrame = 0;
let currentZoneMode = null;
let zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
let detectedObjects = [];
let selectedObject = null;
let isDrawing = false;
let startPoint = null;

// DOM элементы (инициализируются после загрузки DOM)
let uploadArea, fileInput, progressContainer, progressBar, progressText, progressDetails;
let fileList, videoPanel, frameSlider, frameInfo, videoCanvas, overlayCanvas;
let ctx, overlayCtx;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Инициализация обучающего модуля');

    // СНАЧАЛА инициализируем DOM элементы
    initializeDOMElements();

    // ПОТОМ настраиваем обработчики
    setupEventListeners();

    // Загружаем список файлов
    setTimeout(() => {
        loadFileList();
    }, 500);

    // Инициализируем счетчики
    updateZonesCount();
    updateAnnotatedCount();

    console.log('🎉 Инициализация завершена');
});

function initializeDOMElements() {
    console.log('🔄 Инициализация DOM элементов...');

    // Присваиваем значения переменным ИЗ DOM
    uploadArea = document.getElementById('uploadArea');
    fileInput = document.getElementById('fileInput');
    progressContainer = document.getElementById('progressContainer');
    progressBar = document.getElementById('progressBar');
    progressText = document.getElementById('progressText');
    progressDetails = document.getElementById('progressDetails');
    fileList = document.getElementById('fileList');
    videoPanel = document.getElementById('videoPanel');
    frameSlider = document.getElementById('frameSlider');
    frameInfo = document.getElementById('frameInfo');
    videoCanvas = document.getElementById('videoCanvas');
    overlayCanvas = document.getElementById('overlayCanvas');

    // Инициализируем контексты canvas
    if (videoCanvas && overlayCanvas) {
        ctx = videoCanvas.getContext('2d');
        overlayCtx = overlayCanvas.getContext('2d');
        console.log('✅ Canvas контексты инициализированы');
    } else {
        console.error('❌ Canvas элементы не найдены');
    }

    // Проверяем критические элементы
    const criticalElements = [
        {name: 'uploadArea', element: uploadArea},
        {name: 'fileInput', element: fileInput},
        {name: 'fileList', element: fileList},
        {name: 'frameSlider', element: frameSlider}
    ];

    criticalElements.forEach(({name, element}) => {
        if (element) {
            console.log(`✅ ${name} найден`);
        } else {
            console.error(`❌ ${name} НЕ найден`);
        }
    });

    console.log('✅ DOM элементы инициализированы');
}

function setupEventListeners() {
    console.log('🎧 Настройка обработчиков событий...');

    // Проверяем что элементы инициализированы
    if (!uploadArea || !fileInput || !frameSlider) {
        console.error('❌ Критические DOM элементы не найдены');
        console.log('uploadArea:', !!uploadArea, 'fileInput:', !!fileInput, 'frameSlider:', !!frameSlider);
        return;
    }

    // Drag & Drop
    uploadArea.addEventListener('dragenter', handleDragEnter);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => {
        console.log('📁 Клик по области загрузки');
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    // Слайдер кадров
    frameSlider.addEventListener('input', function() {
        loadFrame(parseInt(this.value));
    });

    console.log('✅ Обработчики событий настроены');
}

// === ОБРАБОТЧИКИ DRAG & DROP ===

function handleDragEnter(e) {
    e.preventDefault();
    if (!isUploading) uploadArea.style.borderColor = '#3182ce';
}

function handleDragOver(e) {
    e.preventDefault();
}

function handleDragLeave(e) {
    e.preventDefault();
    if (!uploadArea.contains(e.relatedTarget)) {
        uploadArea.style.borderColor = '#cbd5e0';
    }
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '#cbd5e0';

    if (isUploading) return;

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
        uploadVideoChunked(files[0]);
    } else {
        showStatus('Выберите видео файл', 'error');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && !isUploading) {
        uploadVideoChunked(file);
    }
}

// === ЧАНКОВАЯ ЗАГРУЗКА ===

async function uploadVideoChunked(file) {
    if (isUploading) return;

    isUploading = true;
    uploadArea.classList.add('uploading');
    progressContainer.style.display = 'block';

    try {
        showStatus('Начинается загрузка...', 'info');

        // 1. Инициализация
        const initResponse = await fetch('/api/training/start_upload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filename: file.name,
                file_size: file.size
            })
        });

        const initData = await initResponse.json();
        if (!initData.success) throw new Error(initData.error);

        uploadSessionId = initData.session_id;

        // 2. Загрузка чанками
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const start = chunkIndex * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunk = file.slice(start, end);

            const formData = new FormData();
            formData.append('session_id', uploadSessionId);
            formData.append('chunk_index', chunkIndex);
            formData.append('chunk', chunk);

            const chunkResponse = await fetch('/api/training/upload_chunk', {
                method: 'POST',
                body: formData
            });

            const chunkData = await chunkResponse.json();
            if (!chunkData.success) throw new Error('Ошибка загрузки чанка');

            // Обновляем прогресс
            const progress = chunkData.progress;
            progressBar.style.width = progress + '%';
            progressText.textContent = progress.toFixed(1) + '%';
            progressDetails.textContent =
                `Загружено ${(chunkData.uploaded_size / 1024 / 1024).toFixed(1)} MB из ${(file.size / 1024 / 1024).toFixed(1)} MB`;

            await new Promise(resolve => setTimeout(resolve, 10));
        }

        // 3. Завершение
        const finishResponse = await fetch('/api/training/finish_upload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: uploadSessionId})
        });

        const finishData = await finishResponse.json();
        if (finishData.success) {
            showStatus(finishData.message, 'success');
            totalFrames = finishData.total_frames;
            frameSlider.max = totalFrames - 1;
            frameInfo.textContent = `0 / ${totalFrames}`;

            videoPanel.classList.add('active');
            loadFrame(0);
            loadFileList();
        } else {
            throw new Error(finishData.error);
        }

    } catch (error) {
        showStatus('Ошибка загрузки: ' + error.message, 'error');
    } finally {
        isUploading = false;
        uploadArea.classList.remove('uploading');
        progressContainer.style.display = 'none';
        uploadSessionId = null;
    }
}

// === УПРАВЛЕНИЕ ФАЙЛАМИ ===

function loadFileList() {
    console.log('🔄 Загрузка списка файлов...');

    fetch('/api/training/files')
        .then(response => {
            console.log('📡 Ответ сервера:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('📋 Данные с сервера:', data);

            if (data.files) {
                console.log(`📊 Найдено файлов: ${data.files.length}`);
                renderFileList(data.files);
                updateFileStats(data.files);
            } else {
                console.warn('⚠️ Нет поля files в ответе');
                renderFileList([]);
                updateFileStats([]);
            }
        })
        .catch(error => {
            console.error('❌ Ошибка загрузки списка файлов:', error);
            showStatus('Ошибка загрузки файлов: ' + error.message, 'error');
            renderFileList([]);
            updateFileStats([]);
        });
}

function renderFileList(files) {
    if (!fileList) {
        console.error('❌ Элемент fileList не найден');
        return;
    }

    if (files.length === 0) {
        fileList.innerHTML = '<p style="color: #718096; text-align: center; padding: 1rem;">Видео файлы не найдены.<br>Загрузите видео для обучения</p>';
        return;
    }

    fileList.innerHTML = files.map((file, index) => {
        const sizeText = file.size_gb > 1 ?
            `${file.size_gb} GB` :
            `${file.size_mb} MB`;

        const dateText = new Date(file.modified).toLocaleDateString('ru-RU');

        return `
            <div class="file-item" onclick="selectFile('${file.name}')">
                <div><strong>${file.name}</strong></div>
                <div class="file-info">
                    ${sizeText} • ${file.duration}s • ${dateText}
                </div>
                <div class="file-controls">
                    <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); selectFile('${file.name}')">
                        📂 Открыть
                    </button>
                    <button class="btn btn-sm btn-warning" onclick="event.stopPropagation(); renameFile('${file.name}')">
                        ✏️ 
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteFile('${file.name}')">
                        🗑️
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function updateFileStats(files) {
    const totalFilesElement = document.getElementById('totalFiles');
    const totalSizeElement = document.getElementById('totalSize');

    if (totalFilesElement) {
        totalFilesElement.textContent = files.length;
    }

    if (totalSizeElement) {
        const totalGB = files.reduce((sum, file) => sum + (file.size_gb || 0), 0);
        totalSizeElement.textContent = totalGB.toFixed(1);
    }
}

function selectFile(filename) {
    console.log(`📂 Выбор файла: ${filename}`);

    // Сначала определяем камеру по имени файла
    fetch('/api/training/detect_camera', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({video_name: filename})
    })
    .then(response => response.json())
    .then(cameraData => {
        if (cameraData.success) {
            currentCamera = {
                id: cameraData.camera_id,
                name: cameraData.camera_name
            };
            console.log(`🎥 Определена камера: ${currentCamera.name} (${currentCamera.id})`);
        } else {
            console.warn('⚠️ Не удалось определить камеру, используем по умолчанию');
            currentCamera = {id: 'default', name: 'Камера по умолчанию'};
        }

        // Теперь загружаем видео
        return fetch(`/api/training/files/${filename}/select`, {method: 'POST'});
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus(`${data.message} (${currentCamera.name})`, 'success');
            totalFrames = data.total_frames;
            frameSlider.max = totalFrames - 1;
            frameInfo.textContent = `0 / ${totalFrames}`;

            videoPanel.classList.add('active');
            loadFrame(0);

            // Показываем информацию о камере
            const cameraInfo = document.getElementById('cameraInfo');
            const cameraNameSpan = document.getElementById('cameraName');
            const cameraIdSpan = document.getElementById('cameraId');

            if (cameraInfo && cameraNameSpan && cameraIdSpan) {
                cameraInfo.style.display = 'block';
                cameraNameSpan.textContent = currentCamera.name;
                cameraIdSpan.textContent = currentCamera.id;
            }

            // Выделяем выбранный файл
            document.querySelectorAll('.file-item').forEach(item => {
                item.classList.toggle('selected', item.textContent.includes(filename));
            });

            // Загружаем сохраненные зоны для этой камеры
            loadZonesForCamera(currentCamera.id, currentCamera.name);

        } else {
            showStatus('Ошибка: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка выбора файла: ' + error.message, 'error');
    });
}

function deleteFile(filename) {
    if (confirm(`Удалить файл ${filename}?`)) {
        fetch(`/api/training/files/${filename}`, {method: 'DELETE'})
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Файл удален', 'success');
                    loadFileList();
                } else {
                    showStatus('Ошибка удаления', 'error');
                }
            });
    }
}

function renameFile(filename) {
    const newName = prompt('Новое имя файла:', filename);
    if (newName && newName !== filename) {
        fetch(`/api/training/files/${filename}/rename`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({new_name: newName})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showStatus('Файл переименован', 'success');
                loadFileList();
            } else {
                showStatus('Ошибка переименования', 'error');
            }
        });
    }
}

// === РАБОТА С ВИДЕО ===

function loadFrame(frameIndex) {
    if (frameIndex < 0 || frameIndex >= totalFrames) return;

    showStatus(`Загрузка кадра ${frameIndex}...`, 'info');

    fetch(`/api/training/frame/${frameIndex}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                currentFrame = frameIndex;
                frameSlider.value = frameIndex;
                frameInfo.textContent = `${frameIndex} / ${totalFrames}`;

                const img = new Image();
                img.onload = function() {
                    // Используем оригинальные размеры изображения
                    videoCanvas.width = img.width;
                    videoCanvas.height = img.height;
                    overlayCanvas.width = img.width;
                    overlayCanvas.height = img.height;

                    // Рисуем изображение без масштабирования
                    ctx.drawImage(img, 0, 0);
                    redrawOverlay();

                    setupCanvasEvents();
                };
                img.src = data.frame_data;

                showStatus(`Кадр ${frameIndex} загружен`, 'success');
            } else {
                showStatus('Ошибка загрузки кадра: ' + (data.error || 'неизвестная ошибка'), 'error');
            }
        })
        .catch(error => {
            showStatus('Ошибка сети при загрузке кадра: ' + error.message, 'error');
        });
}

function previousFrame() {
    if (currentFrame > 0) loadFrame(currentFrame - 1);
}

function nextFrame() {
    if (currentFrame < totalFrames - 1) loadFrame(currentFrame + 1);
}

// === РАБОТА С ЗОНАМИ ===

function loadZonesForCamera(cameraId, cameraName) {
    console.log(`🔄 Загрузка зон для камеры: ${cameraName} (${cameraId})`);

    fetch('/api/training/zones/load', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({camera_id: cameraId})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.zones) {
            zones = data.zones;
            console.log('✅ Зоны загружены для камеры:', zones);
            redrawOverlay();
            updateZonesCount();
            showStatus(`Зоны загружены для ${cameraName}`, 'success');
        } else {
            console.log(`ℹ️ Сохраненные зоны для камеры ${cameraName} не найдены`);
            zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
            showStatus(`Зоны для ${cameraName} не найдены - создайте новые`, 'info');
        }
    })
    .catch(error => {
        console.error('❌ Ошибка загрузки зон:', error);
        zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
        showStatus('Ошибка загрузки зон - используем пустые', 'warning');
    });
}

function loadZoneTemplate(templateName) {
    showStatus('Загрузка шаблона зон...', 'info');

    fetch('/api/training/zones/template')
        .then(response => response.json())
        .then(data => {
            if (data.templates && data.templates[templateName]) {
                zones = data.templates[templateName].zones;
                redrawOverlay();
                updateZonesCount();
                showStatus(`Шаблон "${data.templates[templateName].name}" загружен`, 'success');
            } else {
                showStatus('Шаблон не найден', 'error');
            }
        })
        .catch(error => {
            showStatus('Ошибка загрузки шаблона: ' + error.message, 'error');
        });
}

function setZoneMode(mode) {
    // Сбрасываем предыдущий режим если тот же
    if (currentZoneMode === mode) {
        currentZoneMode = null;
        showStatus('Режим разметки выключен', 'info');
    } else {
        currentZoneMode = mode;
        showStatus(`Режим разметки: ${mode} - нарисуйте прямоугольник мышкой`, 'info');
    }

    // Обновляем стили кнопок
    document.querySelectorAll('.zone-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    if (currentZoneMode) {
        const activeBtn = event.target;
        activeBtn.classList.add('active');
    }
}

function saveZones() {
    if (Object.values(zones).every(zone => !zone || (Array.isArray(zone) && zone.length === 0))) {
        showStatus('Сначала нарисуйте хотя бы одну зону', 'warning');
        return;
    }

    showStatus('Сохранение зон...', 'info');

    // Сохраняем зоны для текущей камеры
    fetch('/api/training/zones/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            zones: zones,
            camera_id: currentCamera.id,
            camera_name: currentCamera.name
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus(`Зоны сохранены для ${currentCamera.name}`, 'success');
            updateZonesCount();

            // Автоматическая детекция после сохранения зон
            if (zones.counting_zone) {
                setTimeout(() => {
                    detectObjects();
                }, 500);
            }
        } else {
            showStatus('Ошибка сохранения зон: ' + (data.error || 'неизвестная ошибка'), 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети при сохранении зон: ' + error.message, 'error');
    });
}

function clearZones() {
    if (Object.values(zones).every(zone => !zone || (Array.isArray(zone) && zone.length === 0))) {
        showStatus('Зоны уже очищены', 'info');
        return;
    }

    if (confirm('Удалить все размеченные зоны?')) {
        zones = {entry_zone: null, counting_zone: null, exit_zone: null, gray_zones: []};
        redrawOverlay();
        updateZonesCount();
        showStatus('Все зоны очищены', 'success');
    }
}

function updateZonesCount() {
    let count = 0;
    if (zones.entry_zone) count++;
    if (zones.counting_zone) count++;
    if (zones.exit_zone) count++;
    count += zones.gray_zones.length;

    const zonesCountElement = document.getElementById('zonesCount');
    if (zonesCountElement) {
        zonesCountElement.textContent = count;
    }
}

// === ДЕТЕКЦИЯ ОБЪЕКТОВ ===

function detectObjects() {
    showStatus('Поиск объектов...', 'info');

    fetch('/api/training/detect')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                detectedObjects = data.objects || [];
                renderObjectsList();
                redrawOverlay();
                showStatus(`Найдено объектов: ${detectedObjects.length}`, 'success');
            } else {
                showStatus('Ошибка детекции: ' + (data.error || 'неизвестная ошибка'), 'error');
            }
        })
        .catch(error => {
            showStatus('Ошибка сети при детекции: ' + error.message, 'error');
        });
}

// === АННОТАЦИИ ===

function selectObject(index) {
    selectedObject = index;

    document.querySelectorAll('.object-item').forEach((item, i) => {
        item.classList.toggle('selected', i === index);
    });

    redrawOverlay();

    // Проверяем режим обучения
    if (typeof batchModeEnabled !== 'undefined' && batchModeEnabled) {
        // Пакетный режим - функция определена в batch_training.js
        handleBatchModeObjectSelection(index);
    } else {
        // Обычный режим - показываем форму аннотации
        const obj = detectedObjects[index];
        if (!obj.annotated) {
            document.getElementById('annotationForm').style.display = 'block';
            document.getElementById('productGuid').value = generateGUID();

            // Очищаем предыдущие данные
            document.getElementById('productSku').value = '';
            document.getElementById('productName').value = '';
            document.getElementById('productCategory').value = 'bread';
        } else {
            showStatus(`Объект уже аннотирован: ${obj.annotation_data.product_name}`, 'info');
        }
    }
}

function saveAnnotation() {
    if (selectedObject === null) {
        showStatus('Выберите объект для аннотации', 'error');
        return;
    }

    const obj = detectedObjects[selectedObject];
    const guid = document.getElementById('productGuid').value.trim();
    const sku = document.getElementById('productSku').value.trim();
    const name = document.getElementById('productName').value.trim();
    const category = document.getElementById('productCategory').value;

    if (!sku || !name) {
        showStatus('Заполните SKU и наименование продукта', 'error');
        return;
    }

    const annotation = {
        object_id: obj.id,
        bbox: obj.bbox,
        guid: guid,
        sku_code: sku,
        product_name: name,
        category: category,
        is_validated: true
    };

    showStatus('Сохранение аннотации...', 'info');

    fetch('/api/training/save_annotation', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(annotation)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus(`Аннотация для "${name}" сохранена успешно`, 'success');

            // Помечаем объект как аннотированный
            obj.annotated = true;
            obj.annotation_data = annotation;

            // Обновляем счетчик
            updateAnnotatedCount();

            // Очищаем форму и снимаем выделение
            cancelAnnotation();

            // Перерисовываем с учетом аннотированного объекта
            renderObjectsList();
            redrawOverlay();
        } else {
            showStatus('Ошибка сохранения: ' + (data.error || 'неизвестная ошибка'), 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети при сохранении: ' + error.message, 'error');
    });
}

function cancelAnnotation() {
    document.getElementById('annotationForm').style.display = 'none';
    selectedObject = null;
    redrawOverlay();
}

function renderObjectsList() {
    const list = document.getElementById('objectsList');

    if (detectedObjects.length === 0) {
        list.innerHTML = '<p style="color: #718096; text-align: center; padding: 1rem;">Объекты не найдены.<br>Нажмите "Найти объекты"</p>';
        return;
    }

    list.innerHTML = detectedObjects.map((obj, index) => {
        const statusIcon = obj.annotated ? '✅' : '📝';
        const statusText = obj.annotated ? 'Аннотирован' : 'Требует аннотации';
        const statusClass = obj.annotated ? 'annotated' : 'pending';

        return `<div class="object-item ${statusClass}" onclick="selectObject(${index})">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>${obj.id}</strong>
                <span style="font-size: 12px;">${statusIcon} ${statusText}</span>
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                Позиция: ${obj.bbox.x}, ${obj.bbox.y}<br>
                Размер: ${obj.bbox.width}×${obj.bbox.height}<br>
                Уверенность: ${(obj.confidence * 100).toFixed(1)}%
                ${obj.annotated && obj.annotation_data ? 
                    `<br><strong>SKU:</strong> ${obj.annotation_data.sku_code}<br><strong>Продукт:</strong> ${obj.annotation_data.product_name}` : ''
                }
            </div>
        </div>`;
    }).join('');
}

function updateAnnotatedCount() {
    const currentSessionCount = detectedObjects.filter(obj => obj.annotated).length;

    fetch('/api/training/data')
        .then(response => response.json())
        .then(data => {
            const serverCount = data.total_annotations || 0;
            const annotatedCountElement = document.getElementById('annotatedCount');
            if (annotatedCountElement) {
                annotatedCountElement.textContent = serverCount;
            }
        })
        .catch(error => {
            const annotatedCountElement = document.getElementById('annotatedCount');
            if (annotatedCountElement) {
                annotatedCountElement.textContent = currentSessionCount;
            }
        });
}

// === CANVAS СОБЫТИЯ ===

function setupCanvasEvents() {
    if (!overlayCanvas) return;

    // Удаляем предыдущие обработчики
    overlayCanvas.removeEventListener('mousedown', startDrawing);
    overlayCanvas.removeEventListener('mousemove', draw);
    overlayCanvas.removeEventListener('mouseup', stopDrawing);

    // Добавляем новые обработчики
    overlayCanvas.addEventListener('mousedown', startDrawing);
    overlayCanvas.addEventListener('mousemove', draw);
    overlayCanvas.addEventListener('mouseup', stopDrawing);
}

function startDrawing(e) {
    if (!currentZoneMode) return;

    isDrawing = true;
    const rect = overlayCanvas.getBoundingClientRect();
    startPoint = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function draw(e) {
    if (!isDrawing || !currentZoneMode) return;

    const rect = overlayCanvas.getBoundingClientRect();
    const currentPoint = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };

    redrawOverlay();

    overlayCtx.strokeStyle = getZoneColor(currentZoneMode);
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(
        startPoint.x,
        startPoint.y,
        currentPoint.x - startPoint.x,
        currentPoint.y - startPoint.y
    );
}

function stopDrawing(e) {
    if (!isDrawing || !currentZoneMode) return;

    isDrawing = false;
    const rect = overlayCanvas.getBoundingClientRect();
    const endPoint = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };

    // Проверяем что зона достаточно большая
    const minSize = 20;
    if (Math.abs(endPoint.x - startPoint.x) < minSize || Math.abs(endPoint.y - startPoint.y) < minSize) {
        showStatus('Зона слишком маленькая - нарисуйте больший прямоугольник', 'warning');
        redrawOverlay();
        return;
    }

    const zone = {
        type: currentZoneMode,
        points: [
            startPoint,
            {x: endPoint.x, y: startPoint.y},
            endPoint,
            {x: startPoint.x, y: endPoint.y}
        ]
    };

    if (currentZoneMode === 'gray') {
        zones.gray_zones.push(zone);
    } else {
        zones[currentZoneMode + '_zone'] = zone;
    }

    redrawOverlay();
    updateZonesCount();

    showStatus(`Зона "${currentZoneMode}" создана`, 'success');

    // Автосохранение зон
    setTimeout(() => {
        if (currentCamera && currentCamera.id) {
            fetch('/api/training/zones/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    zones: zones,
                    camera_id: currentCamera.id,
                    camera_name: currentCamera.name
                })
            });
        }
    }, 1000);
}

// === ОТРИСОВКА ===

function redrawOverlay() {
    if (!overlayCtx) return;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Рисуем зоны
    Object.entries(zones).forEach(([key, zone]) => {
        if (key === 'gray_zones') {
            zone.forEach(grayZone => drawZone(grayZone, 'gray'));
        } else if (zone) {
            const zoneType = key.replace('_zone', '');
            drawZone(zone, zoneType);
        }
    });

    // Рисуем объекты
    detectedObjects.forEach((obj, index) => {
        drawObject(obj, index === selectedObject);
    });
}

function drawZone(zone, type) {
    if (!zone.points || zone.points.length < 4) return;

    overlayCtx.strokeStyle = getZoneColor(type);
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([5, 5]);

    overlayCtx.beginPath();
    overlayCtx.moveTo(zone.points[0].x, zone.points[0].y);
    zone.points.forEach(point => {
        overlayCtx.lineTo(point.x, point.y);
    });
    overlayCtx.closePath();
    overlayCtx.stroke();

    overlayCtx.setLineDash([]);
}

function drawObject(obj, isSelected) {
    const bbox = obj.bbox;

    // Цвета для разных состояний объекта
    let strokeColor, fillColor;
    if (obj.annotated) {
        strokeColor = '#22543d';  // Зеленый для аннотированных
        fillColor = 'rgba(34, 84, 61, 0.1)';
    } else if (isSelected) {
        strokeColor = '#3182ce';  // Синий для выбранного
        fillColor = 'rgba(49, 130, 206, 0.1)';
    } else {
        strokeColor = '#e53e3e';  // Красный для обычных
        fillColor = 'rgba(229, 62, 62, 0.05)';
    }

    // Рисуем прямоугольник
    overlayCtx.strokeStyle = strokeColor;
    overlayCtx.fillStyle = fillColor;
    overlayCtx.lineWidth = isSelected ? 3 : 2;

    overlayCtx.fillRect(bbox.x, bbox.y, bbox.width, bbox.height);
    overlayCtx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

    // Подпись объекта
    overlayCtx.fillStyle = strokeColor;
    overlayCtx.font = 'bold 12px Arial';

    let label = obj.id;
    if (obj.annotated && obj.annotation_data) {
        label += ` (${obj.annotation_data.sku_code})`;
    }

    // Фон для текста
    const textMetrics = overlayCtx.measureText(label);
    const textWidth = textMetrics.width + 8;
    const textHeight = 16;

    overlayCtx.fillStyle = strokeColor;
    overlayCtx.fillRect(bbox.x, bbox.y - textHeight - 2, textWidth, textHeight);

    overlayCtx.fillStyle = 'white';
    overlayCtx.fillText(label, bbox.x + 4, bbox.y - 6);

    // Иконка для аннотированных объектов
    if (obj.annotated) {
        overlayCtx.fillStyle = '#22543d';
        overlayCtx.font = 'bold 14px Arial';
        overlayCtx.fillText('✓', bbox.x + bbox.width - 20, bbox.y + 16);
    }
}

function getZoneColor(zoneType) {
    const colors = {
        'entry': '#38a169',
        'counting': '#d69e2e',
        'exit': '#e53e3e',
        'gray': '#718096'
    };
    return colors[zoneType] || '#3182ce';
}

// === УТИЛИТЫ ===

function generateGUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function showStatus(message, type, duration = 4000) {
    // Создаем элемент статуса если его нет
    let statusElement = document.getElementById('statusMessage');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'statusMessage';
        statusElement.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 16px;
            border-radius: 6px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            max-width: 350px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        `;
        document.body.appendChild(statusElement);
    }

    // Устанавливаем цвет в зависимости от типа
    const colors = {
        'success': '#38a169',
        'error': '#e53e3e',
        'warning': '#d69e2e',
        'info': '#3182ce'
    };

    statusElement.style.backgroundColor = colors[type] || colors.info;
    statusElement.textContent = message;
    statusElement.style.display = 'block';
    statusElement.style.transform = 'translateX(0)';

    // Консольный лог для отладки
    const emoji = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    };
    console.log(`${emoji[type] || 'ℹ️'} ${message}`);

    // Автоматически скрываем
    setTimeout(() => {
        statusElement.style.transform = 'translateX(100%)';
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 300);
    }, duration);
}