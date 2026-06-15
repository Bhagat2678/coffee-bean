/* ═══════ Coffee Bean Analyzer — Frontend Logic ═══════ */

// ── DOM Elements ──
const fileFront = document.getElementById('fileFront')
const fileBack = document.getElementById('fileBack')
const dropFront = document.getElementById('drop-front')
const dropBack = document.getElementById('drop-back')
const previewFront = document.getElementById('preview-front')
const previewBack = document.getElementById('preview-back')
const previewFrontImg = document.getElementById('preview-front-img')
const previewBackImg = document.getElementById('preview-back-img')
const analyzeBtn = document.getElementById('analyzeBtn')
const resetBtn = document.getElementById('resetBtn')
const resultsSection = document.getElementById('results')
const annotatedImg = document.getElementById('annotatedImg')
const annotatedImgBack = document.getElementById('annotatedImgBack')
const backImageSection = document.getElementById('backImageSection')
const objectCountEl = document.getElementById('objectCount')
const beanCountEl = document.getElementById('beanCount')
const nonBeanCountEl = document.getElementById('nonBeanCount')
const avgBeanSizeEl = document.getElementById('avgBeanSize')
const avgBeanLengthEl = document.getElementById('avgBeanLength')
const procTimeEl = document.getElementById('procTime')
const colorDistEl = document.getElementById('colorDist')
const detectionsList = document.getElementById('detectionsList')
const toast = document.getElementById('toast')
const downloadBtn = document.getElementById('downloadBtn')
const imageOverlay = document.getElementById('image-overlay')
const sampleWeightInput = document.getElementById('sampleWeight')

// Crop elements
const cropFrontBtn = document.getElementById('cropFrontBtn')
const cropBackBtn = document.getElementById('cropBackBtn')
const cropModal = document.getElementById('cropModal')
const cropImage = document.getElementById('cropImage')
const closeCropModal = document.getElementById('closeCropModal')
const cancelCropBtn = document.getElementById('cancelCropBtn')
const saveCropBtn = document.getElementById('saveCropBtn')

// Grade elements
const gradeBadge = document.getElementById('gradeBadge')
const gradeValue = document.getElementById('gradeValue')
const gradeLabel = document.getElementById('gradeLabel')
const gradeDefects = document.getElementById('gradeDefects')

// Density elements
const densitySection = document.getElementById('densitySection')
const sampleWeightVal = document.getElementById('sampleWeightVal')
const avgDensityVal = document.getElementById('avgDensityVal')

// Size stats elements
const sizeStatsSection = document.getElementById('sizeStatsSection')
const avgLengthVal = document.getElementById('avgLengthVal')
const avgWidthVal = document.getElementById('avgWidthVal')
const lwRatioVal = document.getElementById('lwRatioVal')
const measuredCountVal = document.getElementById('measuredCountVal')

// Screen table elements
const screenSection = document.getElementById('screenSection')
const screenTableBody = document.getElementById('screenTableBody')

// Defect elements
const defectSection = document.getElementById('defectSection')
const defectBreakdown = document.getElementById('defectBreakdown')

// ArcFace elements
const arcfaceSection = document.getElementById('arcfaceSection')
const arcfaceInfo = document.getElementById('arcfaceInfo')

const PLACEHOLDER = '--'
let frontFile = null
let backFile = null
let toastTimer = null

// Crop state variables
let originalFrontFile = null
let originalBackFile = null
let frontManualCrop = false
let backManualCrop = false
let cropperInstance = null
let currentCropSide = null


// ── Utilities ──

function showToast(message, timeout = 3200) {
  if (toastTimer) {
    clearTimeout(toastTimer)
    toastTimer = null
  }
  toast.textContent = message
  toast.classList.remove('hidden')
  toastTimer = setTimeout(() => {
    toast.classList.add('hidden')
  }, timeout)
}

function setAnalyzeState(isLoading) {
  analyzeBtn.disabled = isLoading || !frontFile
  analyzeBtn.querySelector('.btn-text').textContent = isLoading ? 'Analyzing...' : 'Analyze'
  fileFront.disabled = isLoading
  fileBack.disabled = isLoading
  if (imageOverlay) {
    imageOverlay.classList.toggle('hidden', !isLoading)
  }
}

function resetStats() {
  objectCountEl.textContent = PLACEHOLDER
  beanCountEl.textContent = PLACEHOLDER
  nonBeanCountEl.textContent = PLACEHOLDER
  avgBeanSizeEl.textContent = PLACEHOLDER
  avgBeanLengthEl.textContent = PLACEHOLDER
  procTimeEl.textContent = PLACEHOLDER
}

function clearResults() {
  resultsSection.classList.add('hidden')
  detectionsList.innerHTML = ''
  colorDistEl.innerHTML = ''
  annotatedImg.classList.remove('visible')
  annotatedImg.removeAttribute('src')
  if (annotatedImgBack) {
    annotatedImgBack.classList.remove('visible')
    annotatedImgBack.removeAttribute('src')
  }
  if (backImageSection) backImageSection.classList.add('hidden')
  downloadBtn.removeAttribute('href')
  resetStats()
  if (imageOverlay) imageOverlay.classList.add('hidden')

  // Hide new sections
  if (gradeBadge) gradeBadge.classList.add('hidden')
  if (densitySection) densitySection.classList.add('hidden')
  if (sizeStatsSection) sizeStatsSection.classList.add('hidden')
  if (screenSection) screenSection.classList.add('hidden')
  if (defectSection) defectSection.classList.add('hidden')
  if (arcfaceSection) arcfaceSection.classList.add('hidden')
}


// ── Color helpers ──

function colorNameToHex(name) {
  const map = {
    Black: '#242424',
    'Dark Gray': '#4a4a4a',
    Gray: '#8f8f8f',
    'Light Gray': '#c8c8c8',
    White: '#f4f4f4',
    'Dark Brown': '#4a3728',
    Brown: '#7b4d2a',
    'Light Brown': '#a7642a',
    'Orange-Brown': '#b8632d',
    Yellow: '#c9a742',
    Red: '#b03a3a',
    Green: '#2f8a45',
    Blue: '#3d6bb3',
  }
  return map[name] || '#b5b5b5'
}

function createColorSwatch(hex) {
  const swatch = document.createElement('span')
  swatch.className = 'swatch'
  swatch.style.background = hex
  return swatch
}

function humanTypeLabel(type) {
  if (type === 'coffee_bean') return 'Coffee bean'
  if (type === 'non_bean') return 'Non-bean'
  return 'Object'
}


// ── Render functions ──

function renderColorDistribution(colorDistribution = {}) {
  colorDistEl.innerHTML = ''
  const entries = Object.entries(colorDistribution)
  if (!entries.length) return

  const title = document.createElement('div')
  title.className = 'color-title'
  title.textContent = 'Bean Colors'
  colorDistEl.appendChild(title)

  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, count]) => {
      const chip = document.createElement('div')
      chip.className = 'color-chip'
      const label = document.createElement('span')
      label.textContent = `${name}: ${count}`
      chip.appendChild(createColorSwatch(colorNameToHex(name)))
      chip.appendChild(label)
      colorDistEl.appendChild(chip)
    })
}

function renderDetections(detections = []) {
  detectionsList.innerHTML = ''

  if (!detections.length) {
    const empty = document.createElement('div')
    empty.className = 'detection-item'
    empty.textContent = 'No objects detected.'
    detectionsList.appendChild(empty)
    return
  }

  detections.forEach((d, idx) => {
    const type = d.object_type || d.class || 'object'
    const item = document.createElement('div')
    item.className = `detection-item ${type === 'coffee_bean' ? 'is-bean' : 'is-non-bean'}`

    const hex = d.color && d.color.hex ? d.color.hex : '#b5b5b5'
    const colorName = d.color && d.color.name ? d.color.name : 'Unknown'
    const sizeInfo = d.size_mm
    const sizeStr = sizeInfo ? ` | ${sizeInfo.width}×${sizeInfo.height}mm` : ''
    const lengthStr = d.length_mm ? ` | L=${Number(d.length_mm).toFixed(1)}mm` : ''

    const label = document.createElement('span')
    label.className = 'detection-text'
    if (d.defect_type === 'coin') {
      label.textContent = `${idx + 1}. Coin (reference)`
    } else {
      label.textContent = `${idx + 1}. ${humanTypeLabel(type)} | ${colorName}${sizeStr}${lengthStr}`
    }

    item.appendChild(label)
    detectionsList.appendChild(item)
  })
}

function renderGrade(gradeData) {
  if (!gradeData || !gradeBadge) return

  gradeBadge.classList.remove('hidden')
  gradeValue.textContent = gradeData.grade
  gradeLabel.textContent = gradeData.label
  // Defect text hidden temporarily
  // gradeDefects.textContent = `${gradeData.defect_count} defect${gradeData.defect_count !== 1 ? 's' : ''} (${gradeData.defect_percentage}%)`

  gradeValue.className = 'grade-badge'
  const g = gradeData.grade.toLowerCase().replace(/\s+/g, '')
  gradeValue.classList.add(`grade-${g}`)
}

function renderDensity(densityData) {
  if (!densityData || !densitySection) return

  densitySection.classList.remove('hidden')
  sampleWeightVal.textContent = densityData.sample_weight_g || '--'

  if (densityData.avg_weight_per_bean_g != null) {
    avgDensityVal.textContent = densityData.avg_weight_per_bean_g.toFixed(3)
  } else {
    avgDensityVal.textContent = '--'
  }
}

function renderSizeStats(sizeData) {
  if (!sizeData || !sizeStatsSection) return
  if (sizeData.avg_length_mm == null) return

  sizeStatsSection.classList.remove('hidden')
  avgLengthVal.textContent = sizeData.avg_length_mm
  avgWidthVal.textContent = sizeData.avg_width_mm
  lwRatioVal.textContent = sizeData.avg_lw_ratio
  measuredCountVal.textContent = sizeData.bean_count_measured
}

function renderScreenTable(screenData) {
  if (!screenData || !screenSection || !screenTableBody) return
  if (!screenData.length) return

  screenSection.classList.remove('hidden')
  screenTableBody.innerHTML = ''

  const maxPct = Math.max(...screenData.map(r => r.percentage), 1)

  screenData.forEach(row => {
    const tr = document.createElement('tr')

    const tdScreen = document.createElement('td')
    tdScreen.textContent = row.screen
    tdScreen.style.fontWeight = '600'

    const tdAperture = document.createElement('td')
    tdAperture.textContent = row.aperture_mm

    const tdGrade = document.createElement('td')
    tdGrade.textContent = row.africa_india_grade || 'PB'
    tdGrade.style.fontWeight = '600'

    const tdCount = document.createElement('td')
    tdCount.textContent = row.count

    const tdPct = document.createElement('td')
    const barWidth = Math.max(2, (row.percentage / maxPct) * 60)
    tdPct.innerHTML = `<span class="screen-bar" style="width:${barWidth}px"></span>${row.percentage}%`

    tr.appendChild(tdScreen)
    tr.appendChild(tdAperture)
    tr.appendChild(tdGrade)
    tr.appendChild(tdCount)
    tr.appendChild(tdPct)
    screenTableBody.appendChild(tr)
  })
}

function renderDefectBreakdown(gradeData) {
  // Defect rendering temporarily disabled. Preserving original implementation
  // so it can be re-enabled later.
  // if (!gradeData || !defectSection || !defectBreakdown) return
  // const breakdown = gradeData.defect_breakdown || {}
  // const entries = Object.entries(breakdown)
  // if (!entries.length) return
  //
  // defectSection.classList.remove('hidden')
  // defectBreakdown.innerHTML = ''
  //
  // entries.sort((a, b) => b[1] - a[1]).forEach(([type, count]) => {
  //   const chip = document.createElement('div')
  //   const cssClass = type === 'black' ? 'defect-black'
  //     : type === 'broken' ? 'defect-broken'
  //     : type === 'foreign' ? 'defect-foreign'
  //     : 'defect-default'
  //   chip.className = `defect-chip ${cssClass}`
  //
  //   const pct = gradeData.total_beans > 0
  //     ? ((count / gradeData.total_beans) * 100).toFixed(1)
  //     : '0.0'
  //
  //   chip.textContent = `${type}: ${count} (${pct}%)`
  //   defectBreakdown.appendChild(chip)
  // })
}

function renderArcFace(data) {
  if (!arcfaceSection || !arcfaceInfo) return

  if (data.arcface_pair_count && data.arcface_pair_count > 0) {
    arcfaceSection.classList.remove('hidden')
    arcfaceInfo.textContent = `${data.arcface_pair_count} bean pair(s) matched between front and back images. Paired crops saved for training.`
  } else if (data.back_annotated_image_url) {
    arcfaceSection.classList.remove('hidden')
    arcfaceInfo.textContent = 'Front and back images processed. ArcFace matching attempted — add a trained model for better results.'
  }
}


// ── File handling ──

function handleFile(file, side) {
  const allowed = ['image/png', 'image/jpeg']
  if (!allowed.includes(file.type)) {
    showToast('Invalid file type. Use PNG or JPG.')
    return
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Max 10MB.')
    return
  }

  if (side === 'front') {
    frontFile = file
    originalFrontFile = file
    frontManualCrop = false
    previewFrontImg.src = URL.createObjectURL(file)
    previewFront.classList.remove('hidden')
    dropFront.classList.add('hidden')
    if (cropFrontBtn) cropFrontBtn.classList.remove('hidden')
    analyzeBtn.disabled = false
  } else {
    backFile = file
    originalBackFile = file
    backManualCrop = false
    previewBackImg.src = URL.createObjectURL(file)
    previewBack.classList.remove('hidden')
    dropBack.classList.add('hidden')
    if (cropBackBtn) cropBackBtn.classList.remove('hidden')
  }
}

function removeFile(side) {
  if (side === 'front') {
    frontFile = null
    originalFrontFile = null
    frontManualCrop = false
    fileFront.value = ''
    previewFront.classList.add('hidden')
    dropFront.classList.remove('hidden')
    if (cropFrontBtn) cropFrontBtn.classList.add('hidden')
    analyzeBtn.disabled = true
  } else {
    backFile = null
    originalBackFile = null
    backManualCrop = false
    fileBack.value = ''
    previewBack.classList.add('hidden')
    dropBack.classList.remove('hidden')
    if (cropBackBtn) cropBackBtn.classList.add('hidden')
  }
}


// ── Drag & Drop for dual upload ──

function setupDropZone(dropEl, fileInput, side) {
  ;['dragenter', 'dragover'].forEach(evt => {
    dropEl.addEventListener(evt, e => {
      e.preventDefault()
      e.stopPropagation()
      dropEl.classList.add('dragover')
    })
  })

  ;['dragleave', 'drop'].forEach(evt => {
    dropEl.addEventListener(evt, e => {
      e.preventDefault()
      e.stopPropagation()
      dropEl.classList.remove('dragover')
    })
  })

  dropEl.addEventListener('drop', e => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0], side)
    }
  })

  dropEl.addEventListener('click', e => {
    if (e.target === fileInput || e.target.closest('label')) {
      return
    }
    fileInput.click()
  })

  fileInput.addEventListener('change', e => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0], side)
    }
  })
}

setupDropZone(dropFront, fileFront, 'front')
setupDropZone(dropBack, fileBack, 'back')

// Remove buttons
document.querySelectorAll('.remove-btn').forEach(btn => {
  btn.addEventListener('click', e => {
    e.stopPropagation()
    removeFile(btn.dataset.side)
  })
})


// ── Reset ──

resetBtn.addEventListener('click', () => {
  frontFile = null
  backFile = null
  originalFrontFile = null
  originalBackFile = null
  frontManualCrop = false
  backManualCrop = false
  fileFront.value = ''
  fileBack.value = ''
  previewFront.classList.add('hidden')
  previewBack.classList.add('hidden')
  dropFront.classList.remove('hidden')
  dropBack.classList.remove('hidden')
  if (cropFrontBtn) cropFrontBtn.classList.add('hidden')
  if (cropBackBtn) cropBackBtn.classList.add('hidden')
  analyzeBtn.disabled = true
  clearResults()
  sampleWeightInput.value = 350
})


// ── Analyze ──

analyzeBtn.addEventListener('click', async () => {
  if (!frontFile) {
    showToast('Please upload a front-side image.')
    return
  }

  setAnalyzeState(true)
  annotatedImg.classList.remove('visible')
  if (annotatedImgBack) annotatedImgBack.classList.remove('visible')

  const formData = new FormData()
  formData.append('front_image', frontFile)
  formData.append('front_manual_crop', frontManualCrop ? 'true' : 'false')
  if (backFile) {
    formData.append('back_image', backFile)
    formData.append('back_manual_crop', backManualCrop ? 'true' : 'false')
  }

  // Sample weight
  const weight = parseFloat(sampleWeightInput.value) || 350
  formData.append('sample_weight', weight)

  try {
    const response = await fetch('/analyze', { method: 'POST', body: formData })
    const data = await response.json().catch(() => ({}))

    if (!response.ok) {
      showToast(data.error || 'Server error during analysis.')
      return
    }

    // Front annotated image
    annotatedImg.onload = () => {
      if (imageOverlay) imageOverlay.classList.add('hidden')
      annotatedImg.classList.add('visible')
    }
    annotatedImg.src = `${data.annotated_image_url}?t=${Date.now()}`
    downloadBtn.href = data.annotated_image_url

    // Back annotated image (if available)
    if (data.back_annotated_image_url && backImageSection && annotatedImgBack) {
      backImageSection.classList.remove('hidden')
      annotatedImgBack.onload = () => {
        annotatedImgBack.classList.add('visible')
      }
      annotatedImgBack.src = `${data.back_annotated_image_url}?t=${Date.now()}`
    }

    // Core stats
    const objectCount = Number(data.object_count ?? data.bean_count ?? 0)
    const beanCount = Number(data.bean_count ?? 0)
    const nonBeanCount = Number(data.non_bean_count ?? Math.max(0, objectCount - beanCount))

    objectCountEl.textContent = objectCount
    beanCountEl.textContent = beanCount
    nonBeanCountEl.textContent = nonBeanCount
    procTimeEl.textContent = Number(data.processing_time || 0).toFixed(1)

    // Average bean size
    if (data.avg_bean_size_mm) {
      avgBeanSizeEl.textContent = `${data.avg_bean_size_mm.width} × ${data.avg_bean_size_mm.height}`
    } else {
      avgBeanSizeEl.textContent = 'No coin ref'
    }

    // Average bean length
    if (data.avg_bean_length_mm != null) {
      avgBeanLengthEl.textContent = Number(data.avg_bean_length_mm).toFixed(2)
    } else {
      avgBeanLengthEl.textContent = '--'
    }

    // New sections
    renderGrade(data.grade)
    renderDensity(data.density)
    renderSizeStats(data.size_stats)
    renderScreenTable(data.screen_distribution)
    // Defect breakdown rendering disabled for now
    // renderDefectBreakdown(data.grade)
    renderColorDistribution(data.color_distribution || {})
    renderDetections(data.detections || [])
    renderArcFace(data)

    resultsSection.classList.remove('hidden')
    const source = data.detection_source ? ` (${data.detection_source})` : ''
    showToast(`Analysis complete${source}`)
  } catch (err) {
    console.error(err)
    showToast('Network or server error.')
  } finally {
    setAnalyzeState(false)
  }
})


// ── Color picker ──

const colorPickerBtn = document.getElementById('colorPickerBtn')
const pickedColorSection = document.getElementById('pickedColor')
const pickedSwatch = document.getElementById('pickedSwatch')
const pickedColorName = document.getElementById('pickedColorName')
const pickedColorHex = document.getElementById('pickedColorHex')
const imageWrap = document.getElementById('imageWrap')

let isPickerMode = false

function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map(x => {
    const hex = x.toString(16)
    return hex.length === 1 ? '0' + hex : hex
  }).join('').toUpperCase()
}

function rgbToColorName(r, g, b) {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const delta = max - min

  let hue
  if (delta === 0) {
    hue = 0
  } else if (max === r) {
    hue = (60 * (((g - b) / delta) % 6) + 360) % 360
  } else if (max === g) {
    hue = (60 * (((b - r) / delta) + 2) + 360) % 360
  } else {
    hue = (60 * (((r - g) / delta) + 4) + 360) % 360
  }

  const brightness = (max / 255) * 100
  const saturation = (max === 0) ? 0 : (delta / max) * 100

  if (saturation < 15) {
    if (brightness > 90) return 'White'
    if (brightness > 70) return 'Light Gray'
    if (brightness > 40) return 'Gray'
    if (brightness > 15) return 'Dark Gray'
    return 'Black'
  }

  if ((hue >= 0 && hue < 30) || hue >= 330) {
    if (brightness > 60) return 'Orange-Brown'
    if (brightness > 40) return 'Light Brown'
    return 'Dark Brown'
  } else if (hue >= 30 && hue < 60) {
    return 'Yellow'
  } else if (hue >= 60 && hue < 150) {
    return 'Green'
  } else if (hue >= 150 && hue < 270) {
    return 'Blue'
  } else if (hue >= 270 && hue < 330) {
    return 'Red'
  }

  return 'Brown'
}

function handleColorPick() {
  if (!annotatedImg.src || !annotatedImg.offsetParent) {
    showToast('Upload and analyze an image first.')
    return
  }

  isPickerMode = !isPickerMode
  colorPickerBtn.classList.toggle('active', isPickerMode)

  if (isPickerMode) {
    imageWrap.style.cursor = 'crosshair'
    showToast('Click on the image to sample color...', 5000)
  } else {
    imageWrap.style.cursor = 'default'
  }
}

function samplePixelColor(event) {
  if (!isPickerMode || !annotatedImg.src) return

  const canvas = document.createElement('canvas')
  canvas.width = annotatedImg.naturalWidth
  canvas.height = annotatedImg.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(annotatedImg, 0, 0)

  const rect = annotatedImg.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  const scaleX = annotatedImg.naturalWidth / rect.width
  const scaleY = annotatedImg.naturalHeight / rect.height
  const imgX = Math.round(x * scaleX)
  const imgY = Math.round(y * scaleY)

  const imageData = ctx.getImageData(imgX, imgY, 1, 1)
  const [r, g, b] = Array.from(imageData.data.slice(0, 3))
  const hex = rgbToHex(r, g, b)
  const colorName = rgbToColorName(r, g, b)

  pickedSwatch.style.background = hex
  pickedColorName.textContent = colorName
  pickedColorHex.textContent = `RGB(${r}, ${g}, ${b}) · ${hex}`
  pickedColorSection.classList.remove('hidden')

  isPickerMode = false
  colorPickerBtn.classList.remove('active')
  imageWrap.style.cursor = 'default'
  showToast(`Sampled color: ${colorName}`)
}

if (colorPickerBtn) {
  colorPickerBtn.addEventListener('click', handleColorPick)
}
if (annotatedImg) {
  annotatedImg.addEventListener('click', samplePixelColor)
}

resetStats()

// ── Crop Modal Logic ──

function openCropModal(side) {
  currentCropSide = side
  const fileToCrop = side === 'front' ? originalFrontFile : originalBackFile
  if (!fileToCrop) return

  cropImage.src = URL.createObjectURL(fileToCrop)
  cropModal.classList.remove('hidden')

  if (cropperInstance) {
    cropperInstance.destroy()
  }

  cropperInstance = new Cropper(cropImage, {
    viewMode: 1,
    autoCropArea: 0.9,
    responsive: true,
    checkOrientation: false
  })
}

function closeCropModalFn() {
  cropModal.classList.add('hidden')
  if (cropperInstance) {
    cropperInstance.destroy()
    cropperInstance = null
  }
  cropImage.src = ''
}

if (cropFrontBtn) {
  cropFrontBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    openCropModal('front')
  })
}

if (cropBackBtn) {
  cropBackBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    openCropModal('back')
  })
}

if (closeCropModal) closeCropModal.addEventListener('click', closeCropModalFn)
if (cancelCropBtn) cancelCropBtn.addEventListener('click', closeCropModalFn)

if (saveCropBtn) {
  saveCropBtn.addEventListener('click', () => {
    if (!cropperInstance) return

    const canvas = cropperInstance.getCroppedCanvas({
      maxWidth: 2048,
      maxHeight: 2048,
    })

    canvas.toBlob((blob) => {
      if (!blob) return

      const croppedFile = new File([blob], `${currentCropSide}_cropped.jpg`, {
        type: 'image/jpeg',
        lastModified: Date.now()
      })

      if (currentCropSide === 'front') {
        frontFile = croppedFile
        frontManualCrop = true
        previewFrontImg.src = URL.createObjectURL(croppedFile)
      } else {
        backFile = croppedFile
        backManualCrop = true
        previewBackImg.src = URL.createObjectURL(croppedFile)
      }

      closeCropModalFn()
      showToast(`${currentCropSide === 'front' ? 'Front' : 'Back'} image cropped successfully.`)
    }, 'image/jpeg', 0.9)
  })
}
