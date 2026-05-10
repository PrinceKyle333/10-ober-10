/* ─── Settings ──────────────────────────────────────────────────────────────── */
const confidenceSlider = document.getElementById('confidence-slider');
const confidenceValue = document.getElementById('conf-value');
const sizeSlider = document.getElementById('size-slider');
const sizeValue = document.getElementById('size-value');

let confidenceThreshold = 50;
let minImageWidth = 320;

// Update confidence display
confidenceSlider.addEventListener('input', (e) => {
  confidenceThreshold = parseInt(e.target.value);
  confidenceValue.textContent = confidenceThreshold + '%';
});

// Update size display
sizeSlider.addEventListener('input', (e) => {
  minImageWidth = parseInt(e.target.value);
  sizeValue.textContent = minImageWidth;
});

/* ─── Image Mode ────────────────────────────────────────────────────────────── */
const imageUploadArea = document.getElementById('image-upload-area');
const imageInput = document.getElementById('image-input');
const imageIdleState = document.getElementById('image-idle-state');
const imageResultState = document.getElementById('image-result-state');
const imageResultImg = document.getElementById('image-result-img');
const detectionsList = document.getElementById('detections-list');
const imageBtnClear = document.getElementById('image-btn-clear');
const loadingOverlay = document.getElementById('loading-overlay');

let selectedImageFile = null;

// Image drag & drop
imageUploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  imageUploadArea.classList.add('drag-over');
});
imageUploadArea.addEventListener('dragleave', () => imageUploadArea.classList.remove('drag-over'));
imageUploadArea.addEventListener('drop', e => {
  e.preventDefault();
  imageUploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    selectedImageFile = file;
    processImageFile(file);
  }
});

imageUploadArea.addEventListener('click', (e) => {
  if (e.target !== imageInput) imageInput.click();
});

imageInput.addEventListener('change', e => {
  if (e.target.files[0]) {
    selectedImageFile = e.target.files[0];
    processImageFile(selectedImageFile);
  }
});

imageBtnClear.addEventListener('click', () => {
  selectedImageFile = null;
  imageInput.value = '';
  imageIdleState.style.display = 'flex';
  imageResultState.style.display = 'none';
});

async function processImageFile(file) {
  // Check image size
  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    if (img.width < minImageWidth) {
      alert(`Image width (${img.width}px) is below minimum threshold (${minImageWidth}px)`);
      return;
    }
    
    loadingOverlay.classList.add('active');
    imageIdleState.style.display = 'none';
    imageResultState.style.display = 'none';

    const fd = new FormData();
    fd.append('image', file);
    fd.append('confidence', confidenceThreshold / 100);
    fd.append('min_width', minImageWidth);

    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      renderResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
      imageIdleState.style.display = 'flex';
      imageResultState.style.display = 'none';
    } finally {
      loadingOverlay.classList.remove('active');
    }
  };
}

function renderResults(data) {
  imageResultImg.src = `data:image/jpeg;base64,${data.image_b64}`;
  
  const newCount = data.detections.filter(d => {
    const label = (d.label || '').toLowerCase().trim();
    return label.startsWith('new') || label.includes('new 10');
  }).length;
  
  const oldCount = data.detections.filter(d => {
    const label = (d.label || '').toLowerCase().trim();
    return label.startsWith('old') || label.includes('old 10');
  }).length;
  
  const totalCount = newCount + oldCount;
  const COIN_VALUE = 10;

  document.getElementById('det-count').textContent = totalCount;
  document.getElementById('stat-total').textContent = totalCount === 0 ? '—' : totalCount;
  document.getElementById('stat-time').textContent = data.inference_time_ms + ' ms';
  document.getElementById('stat-new').textContent = newCount === 0 ? '—' : newCount;
  document.getElementById('stat-old').textContent = oldCount === 0 ? '—' : oldCount;
  document.getElementById('stat-grandtotal').textContent = `₱${totalCount * COIN_VALUE}`;

  detectionsList.innerHTML = '';
  if (data.detections.length === 0) {
    detectionsList.innerHTML = '<div class="det-placeholder">No coins detected.</div>';
  } else {
    data.detections.forEach((det, i) => {
      const label = (det.label || '').toLowerCase().trim();
      const isNew = label.startsWith('new') || label.includes('new 10');
      const div = document.createElement('div');
      div.className = `det-card ${isNew ? 'new-coin' : 'old-coin'}`;
      div.innerHTML = `
        <div class="det-coin-icon">${isNew ? '✓' : '◯'}</div>
        <div class="det-info">
          <div class="det-label">${det.label}</div>
          <div class="det-conf-row">
            <div class="det-conf-bar">
              <div class="det-conf-fill" style="width:${det.confidence}%"></div>
            </div>
            <div class="det-conf-pct">${det.confidence}%</div>
          </div>
          <div class="det-bbox">bbox [${det.bbox[0]}, ${det.bbox[1]}] → [${det.bbox[2]}, ${det.bbox[3]}]</div>
        </div>
      `;
      detectionsList.appendChild(div);
    });
  }
  
  imageIdleState.style.display = 'none';
  imageResultState.style.display = 'block';
}