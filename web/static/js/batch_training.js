// web/static/js/batch_training.js
// JavaScript для пакетного обучения

// Глобальные переменные для пакетного обучения
let currentBatch = null;
let templateObjects = [];
let autoTrainingActive = false;
let anomalyQueue = [];
let batchModeEnabled = false;

// Переключение режимов обучения
function toggleTrainingMode() {
    const batchPanel = document.getElementById('batchTrainingPanel');
    const regularSection = document.getElementById('regularObjectsSection');
    const toggleBtn = document.getElementById('modeToggleBtn');

    batchModeEnabled = !batchModeEnabled;

    if (batchModeEnabled) {
        // Включаем пакетный режим
        batchPanel.style.display = 'block';
        regularSection.style.display = 'none';
        toggleBtn.textContent = '📝 Ручной режим';
        toggleBtn.className = 'btn btn-secondary';
        showStatus('Включен режим пакетного обучения', 'info');

        // Скрываем форму аннотации если она открыта
        const annotationForm = document.getElementById('annotationForm');
        if (annotationForm) {
            annotationForm.style.display = 'none';
        }

    } else {
        // Включаем ручной режим
        batchPanel.style.display = 'none';
        regularSection.style.display = 'block';
        toggleBtn.textContent = '🏭 Пакетный режим';
        toggleBtn.className = 'btn btn-primary';
        showStatus('Включен режим ручного обучения', 'info');
    }

    // Перерисовываем объекты с учетом нового режима
    if (typeof detectedObjects !== 'undefined' && detectedObjects.length > 0) {
        renderObjectsList();
        redrawOverlay();
    }
}

// Обработчик выбора объекта в пакетном режиме
function handleBatchModeObjectSelection(index) {
    const obj = detectedObjects[index];

    if (!obj) {
        showStatus('Объект не найден', 'error');
        return;
    }

    if (currentBatch && !autoTrainingActive) {
        // Добавляем к эталону
        addToTemplate(index);
    } else if (autoTrainingActive) {
        showObjectClassification(obj);
    } else {
        showStatus('Сначала создайте партию', 'warning');
    }
}

// === СОЗДАНИЕ ПАРТИИ ===

function createBatch() {
    const batchData = {
        product_name: document.getElementById('batchProductName').value.trim(),
        sku_code: document.getElementById('batchSku').value.trim(),
        category: document.getElementById('batchCategory').value,
        batch_size_estimate: parseInt(document.getElementById('batchSize').value),
        operator: document.getElementById('batchOperator').value.trim() || 'Система'
    };

    if (!batchData.product_name || !batchData.sku_code) {
        showStatus('Заполните название продукта и SKU', 'error');
        return;
    }

    fetch('/api/training/batch/create', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(batchData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentBatch = data.batch_id;
            showStatus(data.message, 'success');

            // Показываем панели
            document.getElementById('batchCreationForm').style.display = 'none';
            document.getElementById('currentBatchInfo').style.display = 'block';
            document.getElementById('templateSection').style.display = 'block';

            // Обновляем информацию о партии
            document.getElementById('batchId').textContent = data.batch_id;
            document.getElementById('batchProduct').textContent = batchData.product_name;
            document.getElementById('batchStatus').textContent = 'created';
            document.getElementById('batchStatus').className = 'status-badge status-created';

        } else {
            showStatus('Ошибка создания партии: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети: ' + error.message, 'error');
    });
}

// === РАБОТА С ЭТАЛОНОМ ===

function addToTemplate(objectIndex) {
    if (objectIndex < 0 || objectIndex >= detectedObjects.length) {
        showStatus('Неверный индекс объекта', 'error');
        return;
    }

    const obj = detectedObjects[objectIndex];

    if (templateObjects.find(t => t.id === obj.id)) {
        showStatus(`Объект ${obj.id} уже в эталоне`, 'warning');
        return;
    }

    // Создаем копию объекта для эталона
    const templateObj = {
        id: obj.id,
        bbox: {...obj.bbox},
        center: {...obj.center},
        confidence: obj.confidence,
        area: obj.area || (obj.bbox.width * obj.bbox.height)
    };

    templateObjects.push(templateObj);
    renderTemplateObjects();

    showStatus(`Объект ${obj.id} добавлен к эталону (${templateObjects.length})`, 'success');
}

function renderTemplateObjects() {
    const container = document.getElementById('selectedTemplateObjects');

    if (!container) return;

    if (templateObjects.length === 0) {
        container.innerHTML = '<p style="color: #666; font-size: 11px;">Эталонные объекты не выбраны</p>';
        return;
    }

    container.innerHTML = templateObjects.map((obj, index) => `
        <div class="template-object-item">
            <span>${obj.id} (${obj.bbox.width}×${obj.bbox.height})</span>
            <button class="btn btn-sm btn-danger" onclick="removeFromTemplate(${index})" style="font-size: 9px; padding: 1px 4px;">×</button>
        </div>
    `).join('');
}

function removeFromTemplate(index) {
    templateObjects.splice(index, 1);
    renderTemplateObjects();

    if (typeof redrawOverlay === 'function') {
        redrawOverlay();
    }
}

function clearTemplate() {
    templateObjects = [];
    renderTemplateObjects();

    if (typeof redrawOverlay === 'function') {
        redrawOverlay();
    }

    showStatus('Эталон очищен', 'info');
}

function setTemplate() {
    if (templateObjects.length < 2) {
        showStatus('Выберите минимум 2 эталонных объекта', 'warning');
        return;
    }

    fetch('/api/training/batch/set_template', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({template_objects: templateObjects})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus(data.message, 'success');

            // Обновляем статус
            document.getElementById('batchStatus').textContent = 'template_ready';
            document.getElementById('batchStatus').className = 'status-badge status-template_ready';

            // Показываем секцию автообучения
            document.getElementById('autoTrainingSection').style.display = 'block';
            document.getElementById('templateSection').style.display = 'none';

        } else {
            showStatus('Ошибка установки эталона: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети: ' + error.message, 'error');
    });
}

// === АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ ===

function startAutoTraining() {
    fetch('/api/training/batch/start_auto', {method: 'POST'})
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            autoTrainingActive = true;
            showStatus(data.message, 'success');

            // Обновляем UI
            document.getElementById('startAutoBtn').style.display = 'none';
            document.getElementById('stopAutoBtn').style.display = 'block';
            document.getElementById('trainingProgress').style.display = 'block';
            document.getElementById('batchStatus').textContent = 'auto_training';
            document.getElementById('batchStatus').className = 'status-badge status-auto_training';

        } else {
            showStatus('Ошибка запуска автообучения: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети: ' + error.message, 'error');
    });
}

function stopAutoTraining() {
    fetch('/api/training/batch/stop', {method: 'POST'})
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            autoTrainingActive = false;
            showStatus(data.message, 'info');

            // Обновляем UI
            document.getElementById('startAutoBtn').style.display = 'block';
            document.getElementById('stopAutoBtn').style.display = 'none';

        } else {
            showStatus('Ошибка остановки: ' + data.error, 'error');
        }
    });
}

function processCurrentFrame() {
    if (!autoTrainingActive) {
        showStatus('Сначала запустите автообучение', 'warning');
        return;
    }

    showStatus('Обработка кадра...', 'info');

    fetch('/api/training/batch/process_frame', {method: 'POST'})
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const results = data.results;

            // Обновляем прогресс
            updateTrainingProgress(results);

            // Обновляем объекты с классификацией
            if (data.detected_objects) {
                // Добавляем классификацию к объектам
                data.detected_objects.forEach((obj, index) => {
                    if (results.classifications && results.classifications[index]) {
                        obj.classification = results.classifications[index];
                    }
                });

                // Обновляем глобальный массив
                if (typeof detectedObjects !== 'undefined') {
                    detectedObjects = data.detected_objects;
                    renderObjectsList();
                    redrawOverlay();
                }
            }

            if (results.stop_required) {
                showStatus(`Обнаружены аномалии! Требуется вмешательство оператора`, 'warning');
                loadAnomalyQueue();
            } else if (results.processed > 0) {
                showStatus(`Обработано объектов: ${results.processed} (хороших: ${results.good}, брак: ${results.defects})`, 'success');
            } else {
                showStatus('На кадре нет объектов для обработки', 'info');
            }

        } else {
            showStatus('Ошибка обработки: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети: ' + error.message, 'error');
    });
}

// === СТАТИСТИКА И ПРОГРЕСС ===

function updateTrainingProgress(results) {
    // Получаем общую статистику партии
    fetch('/api/training/batch/statistics')
    .then(response => response.json())
    .then(data => {
        if (data.success && data.statistics) {
            const stats = data.statistics;

            const processedElement = document.getElementById('processedCount');
            const goodElement = document.getElementById('goodCount');
            const defectElement = document.getElementById('defectCount');
            const anomalyElement = document.getElementById('anomalyCount');

            if (processedElement) processedElement.textContent = stats.processed_objects || 0;
            if (goodElement) goodElement.textContent = stats.good_objects || 0;
            if (defectElement) defectElement.textContent = stats.defect_objects || 0;
            if (anomalyElement) anomalyElement.textContent = stats.anomaly_objects || 0;
        }
    })
    .catch(error => {
        console.warn('Не удалось обновить статистику:', error);
    });
}

// === РАБОТА С АНОМАЛИЯМИ ===

function loadAnomalyQueue() {
    fetch('/api/training/batch/anomalies')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            anomalyQueue = data.anomalies;
            renderAnomalyQueue();

            if (anomalyQueue.length > 0) {
                document.getElementById('anomalyQueueSection').style.display = 'block';
                document.getElementById('anomalyQueueCount').textContent = anomalyQueue.length;
            }
        }
    })
    .catch(error => {
        console.error('Ошибка загрузки аномалий:', error);
    });
}

function renderAnomalyQueue() {
    const container = document.getElementById('anomalyQueue');

    if (!container) return;

    if (anomalyQueue.length === 0) {
        container.innerHTML = '<p style="color: #666; font-size: 11px;">Аномалий нет</p>';
        return;
    }

    container.innerHTML = anomalyQueue.map(anomaly => `
        <div class="anomaly-item">
            <div class="anomaly-item-header">
                <span class="anomaly-item-id">${anomaly.object.id}</span>
                <span class="anomaly-item-similarity">Схожесть: ${(anomaly.classification.similarity * 100).toFixed(1)}%</span>
            </div>
            <div style="font-size: 10px; color: #666;">
                Размер: ${anomaly.object.bbox.width}×${anomaly.object.bbox.height}<br>
                Причина: ${anomaly.classification.reason || 'Неизвестный тип объекта'}
            </div>
            <select class="defect-category-select" id="defectCategory_${anomaly.id}" style="display: none;">
                <option value="merged">Слипшиеся</option>
                <option value="deformed">Деформированные</option>
                <option value="undercooked">Недопеченные</option>
                <option value="overcooked">Перепеченные</option>
                <option value="size_anomaly">Неправильный размер</option>
                <option value="foreign_object">Посторонний объект</option>
            </select>
            <div class="anomaly-actions">
                <button class="btn btn-success" onclick="resolveAnomaly(${anomaly.id}, 'add_to_good')">✅ Хороший</button>
                <button class="btn btn-danger" onclick="showDefectCategories(${anomaly.id})">❌ Брак</button>
                <button class="btn btn-secondary" onclick="resolveAnomaly(${anomaly.id}, 'ignore')">⏭️ Игнор</button>
            </div>
        </div>
    `).join('');
}

function showDefectCategories(anomalyId) {
    const select = document.getElementById(`defectCategory_${anomalyId}`);
    if (select) {
        select.style.display = 'block';
        select.onchange = () => resolveAnomaly(anomalyId, 'mark_as_defect', select.value);
    }
}

function resolveAnomaly(anomalyId, action, defectCategory = null) {
    const resolution = {action: action};
    if (defectCategory) {
        resolution.defect_category = defectCategory;
    }

    fetch('/api/training/batch/resolve_anomaly', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            anomaly_id: anomalyId,
            resolution: resolution
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus('Аномалия разрешена', 'success');
            loadAnomalyQueue(); // Перезагружаем очередь
            updateTrainingProgress(); // Обновляем статистику
        } else {
            showStatus('Ошибка разрешения аномалии: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStatus('Ошибка сети: ' + error.message, 'error');
    });
}

// === ОТОБРАЖЕНИЕ ОБЪЕКТОВ В ПАКЕТНОМ РЕЖИМЕ ===

function showObjectClassification(obj) {
    // Показываем классификацию объекта
    let message = `Объект ${obj.id}:\n`;
    if (obj.classification) {
        if (obj.classification.type === 'good') {
            message += `✅ Хороший (схожесть: ${(obj.classification.similarity * 100).toFixed(1)}%)`;
        } else if (obj.classification.type === 'defect') {
            message += `❌ Брак: ${obj.classification.defect_category}`;
        } else if (obj.classification.type === 'anomaly') {
            message += `⚠️ Аномалия (схожесть: ${(obj.classification.similarity * 100).toFixed(1)}%)`;
        }
    } else {
        message += 'Классификация не выполнена';
    }

    showStatus(message, 'info', 6000);
}

// Переопределяем функции отрисовки для пакетного режима
function renderObjectsListBatch() {
    if (!batchModeEnabled || typeof detectedObjects === 'undefined') {
        return;
    }

    const list = document.getElementById('objectsList');
    if (!list) return;

    if (detectedObjects.length === 0) {
        list.innerHTML = '<p style="color: #718096; text-align: center; padding: 1rem;">Объекты не найдены.<br>Нажмите "Найти объекты"</p>';
        return;
    }

    list.innerHTML = detectedObjects.map((obj, index) => {
        let statusIcon, statusText, statusClass;

        // В пакетном режиме показываем классификацию
        if (templateObjects.find(t => t.id === obj.id)) {
            statusIcon = '🎯';
            statusText = 'Эталон';
            statusClass = 'template';
        } else if (obj.classification) {
            switch (obj.classification.type) {
                case 'good':
                    statusIcon = '✅';
                    statusText = `Хороший (${(obj.classification.similarity * 100).toFixed(0)}%)`;
                    statusClass = 'good';
                    break;
                case 'defect':
                    statusIcon = '❌';
                    statusText = `Брак: ${obj.classification.defect_category}`;
                    statusClass = 'defect';
                    break;
                case 'anomaly':
                    statusIcon = '⚠️';
                    statusText = `Аномалия (${(obj.classification.similarity * 100).toFixed(0)}%)`;
                    statusClass = 'anomaly';
                    break;
                default:
                    statusIcon = '❓';
                    statusText = 'Не классифицирован';
                    statusClass = 'unknown';
            }
        } else {
            statusIcon = '📝';
            statusText = 'Ожидает классификации';
            statusClass = 'pending';
        }

        return `<div class="object-item ${statusClass}" onclick="selectObject(${index})">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>${obj.id}</strong>
                <span style="font-size: 12px;">${statusIcon} ${statusText}</span>
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                Позиция: ${obj.bbox.x}, ${obj.bbox.y}<br>
                Размер: ${obj.bbox.width}×${obj.bbox.height}<br>
                Уверенность: ${(obj.confidence * 100).toFixed(1)}%
                ${obj.classification ? 
                    `<br><strong>Схожесть:</strong> ${(obj.classification.similarity * 100).toFixed(1)}%` : ''
                }
            </div>
        </div>`;
    }).join('');
}

function drawObjectBatch(obj, isSelected) {
    if (!batchModeEnabled) return;

    const bbox = obj.bbox;

    // Определяем цвет в зависимости от классификации в пакетном режиме
    let strokeColor, fillColor;

    if (templateObjects.find(t => t.id === obj.id)) {
        // Эталонные объекты
        strokeColor = '#2b6cb0';  // Синий для эталона
        fillColor = 'rgba(43, 108, 176, 0.1)';
    } else if (obj.classification) {
        switch (obj.classification.type) {
            case 'good':
                strokeColor = '#22543d';  // Зеленый для хороших
                fillColor = 'rgba(34, 84, 61, 0.1)';
                break;
            case 'defect':
                strokeColor = '#e53e3e';  // Красный для brака
                fillColor = 'rgba(229, 62, 62, 0.1)';
                break;
            case 'anomaly':
                strokeColor = '#d69e2e';  // Оранжевый для аномалий
                fillColor = 'rgba(214, 158, 46, 0.1)';
                break;
            default:
                strokeColor = '#718096';  // Серый для неклассифицированных
                fillColor = 'rgba(113, 128, 150, 0.05)';
        }
    } else {
        strokeColor = '#718096';  // Серый для неклассифицированных
        fillColor = 'rgba(113, 128, 150, 0.05)';
    }

    if (isSelected) {
        strokeColor = '#3182ce';  // Синий для выбранного
        fillColor = 'rgba(49, 130, 206, 0.2)';
    }

    // Рисуем прямоугольник
    if (typeof overlayCtx !== 'undefined') {
        overlayCtx.strokeStyle = strokeColor;
        overlayCtx.fillStyle = fillColor;
        overlayCtx.lineWidth = isSelected ? 3 : 2;

        overlayCtx.fillRect(bbox.x, bbox.y, bbox.width, bbox.height);
        overlayCtx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

        // Подпись объекта в пакетном режиме
        overlayCtx.fillStyle = strokeColor;
        overlayCtx.font = 'bold 12px Arial';

        let label = obj.id;

        // В пакетном режиме показываем классификацию
        if (templateObjects.find(t => t.id === obj.id)) {
            label += ' (эталон)';
        } else if (obj.classification) {
            switch (obj.classification.type) {
                case 'good':
                    label += ` ✅ ${(obj.classification.similarity * 100).toFixed(0)}%`;
                    break;
                case 'defect':
                    label += ` ❌ ${obj.classification.defect_category}`;
                    break;
                case 'anomaly':
                    label += ` ⚠️ ${(obj.classification.similarity * 100).toFixed(0)}%`;
                    break;
            }
        }

        // Фон для текста
        const textMetrics = overlayCtx.measureText(label);
        const textWidth = textMetrics.width + 8;
        const textHeight = 16;

        overlayCtx.fillStyle = strokeColor;
        overlayCtx.fillRect(bbox.x, bbox.y - textHeight - 2, textWidth, textHeight);

        overlayCtx.fillStyle = 'white';
        overlayCtx.fillText(label, bbox.x + 4, bbox.y - 6);
    }
}

// === НАСТРОЙКИ ===

// Обновление порога схожести
document.addEventListener('DOMContentLoaded', function() {
    const thresholdSlider = document.getElementById('similarityThreshold');
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', function() {
            const thresholdValue = document.getElementById('thresholdValue');
            if (thresholdValue) {
                thresholdValue.textContent = this.value;
            }

            // Отправляем на сервер
            fetch('/api/training/batch/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({similarity_threshold: parseFloat(this.value)})
            })
            .catch(error => {
                console.warn('Ошибка обновления настроек:', error);
            });
        });
    }
});

// Переопределяем функции для работы в пакетном режиме
if (typeof window !== 'undefined') {
    // Сохраняем оригинальные функции
    const originalRenderObjectsList = window.renderObjectsList;
    const originalDrawObject = window.drawObject;

    // Переопределяем с учетом режима
    window.renderObjectsList = function() {
        if (batchModeEnabled) {
            renderObjectsListBatch();
        } else if (originalRenderObjectsList) {
            originalRenderObjectsList();
        }
    };

    window.drawObject = function(obj, isSelected) {
        if (batchModeEnabled) {
            drawObjectBatch(obj, isSelected);
        } else if (originalDrawObject) {
            originalDrawObject(obj, isSelected);
        }
    };
}