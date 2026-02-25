const fileElem = document.getElementById('fileElem')
const dropArea = document.getElementById('drop-area')
const preview = document.getElementById('preview')
const previewImg = document.getElementById('preview-img')
const analyzeBtn = document.getElementById('analyzeBtn')
const resetBtn = document.getElementById('resetBtn')
const resultsSection = document.getElementById('results')
const annotatedImg = document.getElementById('annotatedImg')
const objectCountEl = document.getElementById('objectCount')
const beanCountEl = document.getElementById('beanCount')
const nonBeanCountEl = document.getElementById('nonBeanCount')
const avgBeanSizeEl = document.getElementById('avgBeanSize')
const procTimeEl = document.getElementById('procTime')
const colorDistEl = document.getElementById('colorDist')
const detectionsList = document.getElementById('detectionsList')
const toast = document.getElementById('toast')
const downloadBtn = document.getElementById('downloadBtn')
const imageOverlay = document.getElementById('image-overlay')

const PLACEHOLDER = '--'
let currentFile = null
let toastTimer = null



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
  analyzeBtn.disabled = isLoading || !currentFile
  analyzeBtn.querySelector('.btn-text').textContent = isLoading ? 'Analyzing...' : 'Upload and Analyze'
  fileElem.disabled = isLoading
  if (imageOverlay) {
    imageOverlay.classList.toggle('hidden', !isLoading)
  }
}

function resetStats() {
  objectCountEl.textContent = PLACEHOLDER
  beanCountEl.textContent = PLACEHOLDER
  nonBeanCountEl.textContent = PLACEHOLDER
  avgBeanSizeEl.textContent = PLACEHOLDER
  procTimeEl.textContent = PLACEHOLDER
}

function clearResults() {
  resultsSection.classList.add('hidden')
  detectionsList.innerHTML = ''
  colorDistEl.innerHTML = ''
  annotatedImg.classList.remove('visible')
  annotatedImg.removeAttribute('src')
  downloadBtn.removeAttribute('href')
  resetStats()
  if (imageOverlay) {
    imageOverlay.classList.add('hidden')
  }
}

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
  if (type === 'coffee_bean') {
    return 'Coffee bean'
  }
  if (type === 'non_bean') {
    return 'Non-bean'
  }
  return 'Object'
}

function renderColorDistribution(colorDistribution = {}) {
  colorDistEl.innerHTML = ''
  const entries = Object.entries(colorDistribution)
  if (!entries.length) {
    return
  }

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
    const confidence = Number(d.confidence || 0).toFixed(3)
    const sizeInfo = d.size_mm
    const sizeStr = sizeInfo ? ` | ${sizeInfo.width}×${sizeInfo.height}mm` : ''

    const label = document.createElement('span')
    label.className = 'detection-text'
    if (d.defect_type === 'coin') {
      label.textContent = `${idx + 1}. Coin (reference)`
    } else {
      label.textContent = `${idx + 1}. ${humanTypeLabel(type)} | ${colorName}${sizeStr}`
    }

    item.appendChild(createColorSwatch(hex))
    item.appendChild(label)
    detectionsList.appendChild(item)
  })
}

function handleFile(file) {
  const allowed = ['image/png', 'image/jpeg']
  if (!allowed.includes(file.type)) {
    showToast('Invalid file type. Use PNG or JPG.')
    return
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Max 10MB.')
    return
  }

  currentFile = file
  previewImg.src = URL.createObjectURL(file)
  preview.classList.remove('hidden')
  analyzeBtn.disabled = false
}

;['dragenter', 'dragover'].forEach(evt => {
  dropArea.addEventListener(evt, e => {
    e.preventDefault()
    e.stopPropagation()
    dropArea.classList.add('dragover')
  })
})

  ;['dragleave', 'drop'].forEach(evt => {
    dropArea.addEventListener(evt, e => {
      e.preventDefault()
      e.stopPropagation()
      dropArea.classList.remove('dragover')
    })
  })

dropArea.addEventListener('drop', e => {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0])
  }
})

dropArea.addEventListener('click', () => fileElem.click())

fileElem.addEventListener('change', e => {
  if (e.target.files && e.target.files[0]) {
    handleFile(e.target.files[0])
  }
})

resetBtn.addEventListener('click', () => {
  currentFile = null
  fileElem.value = ''
  preview.classList.add('hidden')
  analyzeBtn.disabled = true
  clearResults()
})

analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) {
    showToast('No image selected.')
    return
  }

  setAnalyzeState(true)
  annotatedImg.classList.remove('visible')

  const formData = new FormData()
  formData.append('image', currentFile)

  // Add confidence threshold from slider
  const confThreshold = document.getElementById('confThreshold')
  if (confThreshold) {
    formData.append('confidence_threshold', confThreshold.value)
  }

  try {
    const response = await fetch('/analyze', { method: 'POST', body: formData })
    const data = await response.json().catch(() => ({}))

    if (!response.ok) {
      showToast(data.error || 'Server error during analysis.')
      return
    }

    annotatedImg.onload = () => {
      if (imageOverlay) {
        imageOverlay.classList.add('hidden')
      }
      annotatedImg.classList.add('visible')
    }

    annotatedImg.src = `${data.annotated_image_url}?t=${Date.now()}`
    downloadBtn.href = data.annotated_image_url

    const objectCount = Number(data.object_count ?? data.bean_count ?? 0)
    const beanCount = Number(data.bean_count ?? 0)
    const nonBeanCount = Number(data.non_bean_count ?? Math.max(0, objectCount - beanCount))

    objectCountEl.textContent = objectCount
    beanCountEl.textContent = beanCount
    nonBeanCountEl.textContent = nonBeanCount
    procTimeEl.textContent = Number(data.processing_time || 0).toFixed(1)

    // Display average bean size
    if (data.avg_bean_size_mm) {
      avgBeanSizeEl.textContent = `${data.avg_bean_size_mm.width} × ${data.avg_bean_size_mm.height}`
    } else {
      avgBeanSizeEl.textContent = 'No coin ref'
    }

    renderColorDistribution(data.color_distribution || {})
    renderDetections(data.detections || [])

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

// Color picker functionality
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
  // RGB to color name mapping for coffee beans
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const delta = max - min

  // Calculate hue (0-360)
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

  // Calculate brightness (0-100)
  const brightness = (max / 255) * 100

  // Calculate saturation (0-100)  
  const saturation = (max === 0) ? 0 : (delta / max) * 100

  // Use brightness to distinguish gray tones
  if (saturation < 15) {
    if (brightness > 90) return 'White'
    if (brightness > 70) return 'Light Gray'
    if (brightness > 40) return 'Gray'
    if (brightness > 15) return 'Dark Gray'
    return 'Black'
  }

  // Color-based mapping (coffee bean focused)
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
    imageWrap.style.cursor = 'eyedropper'
    showToast('Click on the image to sample color...', 5000)
  } else {
    imageWrap.style.cursor = 'default'
  }
}

function samplePixelColor(event) {
  if (!isPickerMode || !annotatedImg.src) return

  // Create canvas from image
  const canvas = document.createElement('canvas')
  canvas.width = annotatedImg.naturalWidth
  canvas.height = annotatedImg.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(annotatedImg, 0, 0)

  // Get click position relative to displayed image
  const rect = annotatedImg.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  // Scale to natural image dimensions
  const scaleX = annotatedImg.naturalWidth / rect.width
  const scaleY = annotatedImg.naturalHeight / rect.height
  const imgX = Math.round(x * scaleX)
  const imgY = Math.round(y * scaleY)

  // Sample pixel
  const imageData = ctx.getImageData(imgX, imgY, 1, 1)
  const [r, g, b] = Array.from(imageData.data.slice(0, 3))
  const hex = rgbToHex(r, g, b)
  const colorName = rgbToColorName(r, g, b)

  // Display result
  pickedSwatch.style.background = hex
  pickedColorName.textContent = colorName
  pickedColorHex.textContent = `RGB(${r}, ${g}, ${b}) · ${hex}`
  pickedColorSection.classList.remove('hidden')

  // Exit picker mode
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
