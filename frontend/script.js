class BuildingDetector {
  constructor() {
    this.imageFile = null;
    this.imageFileT1 = null;
    this.imageFileT2 = null;
    this.backendUrl = 'http://localhost:5000';
    this.gpsBounds = null;
    this.currentSessionId = null;
    this.currentMode = 'temporal';
    this.currentResults = null;
    this.init();
  }

  init() {
    console.log('🚀 Initializing BuildingDetector with backend:', this.backendUrl);
    
    this.setupSingleMode();
    this.setupTemporalMode();
    this.setupModeSwitching();
    this.setupGPSToggles();
    this.setupThresholdSlider();
    
    console.log('✅ BuildingDetector initialization complete');
  }

  setupGPSToggles() {
    const gpsToggle = document.getElementById('gpsToggle');
    if (gpsToggle) {
      gpsToggle.addEventListener('change', (e) => {
        document.getElementById('gpsInputs').style.display = e.target.checked ? 'block' : 'none';
      });
    }

    const gpsToggleTemporal = document.getElementById('gpsToggleTemporal');
    if (gpsToggleTemporal) {
      gpsToggleTemporal.addEventListener('change', (e) => {
        document.getElementById('gpsInputsTemporal').style.display = e.target.checked ? 'block' : 'none';
      });
    }
  }

  setupThresholdSlider() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
        document.getElementById('thresholdValue').textContent = e.target.value;
      });
    }
  }

  setupSingleMode() {
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const processBtn = document.getElementById('processBtn');

    if (uploadZone && imageInput) {
      uploadZone.addEventListener('click', () => imageInput.click());
      
      imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFile(e.target.files[0], 'single');
        }
      });
      
      this.setupDragDrop(uploadZone, (file) => this.handleFile(file, 'single'));
    }

    if (processBtn) {
      processBtn.addEventListener('click', () => this.processImage());
    }
  }

  setupTemporalMode() {
    const uploadZoneT1 = document.getElementById('uploadZoneT1');
    const imageInputT1 = document.getElementById('imageInputT1');
    const uploadZoneT2 = document.getElementById('uploadZoneT2');
    const imageInputT2 = document.getElementById('imageInputT2');

    if (uploadZoneT1 && imageInputT1) {
      uploadZoneT1.addEventListener('click', () => imageInputT1.click());
      imageInputT1.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFile(e.target.files[0], 't1');
        }
      });
      this.setupDragDrop(uploadZoneT1, (file) => this.handleFile(file, 't1'));
    }

    if (uploadZoneT2 && imageInputT2) {
      uploadZoneT2.addEventListener('click', () => imageInputT2.click());
      imageInputT2.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFile(e.target.files[0], 't2');
        }
      });
      this.setupDragDrop(uploadZoneT2, (file) => this.handleFile(file, 't2'));
    }

    const processBtnTemporal = document.getElementById('processBtnTemporal');
    if (processBtnTemporal) {
      processBtnTemporal.addEventListener('click', () => this.processTwoTemporal());
    }
  }

  setupDragDrop(element, callback) {
    element.addEventListener('dragover', (e) => {
      e.preventDefault();
      element.classList.add('drag-over');
    });
    
    element.addEventListener('dragleave', () => {
      element.classList.remove('drag-over');
    });
    
    element.addEventListener('drop', (e) => {
      e.preventDefault();
      element.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        callback(e.dataTransfer.files[0]);
      }
    });
  }

  setupModeSwitching() {
    const modeSingle = document.getElementById('modeSingle');
    const modeTemporal = document.getElementById('modeTemporal');
    
    if (modeSingle) {
      modeSingle.addEventListener('click', () => this.switchMode('single'));
    }
    
    if (modeTemporal) {
      modeTemporal.addEventListener('click', () => this.switchMode('temporal'));
    }
  }

  switchMode(mode) {
    console.log('🔄 Switching mode to:', mode);
    this.currentMode = mode;
    
    const singleCard = document.getElementById('singleImageCard');
    const temporalCard = document.getElementById('temporalCard');
    const modeSingle = document.getElementById('modeSingle');
    const modeTemporal = document.getElementById('modeTemporal');

    this.hideResults();

    if (mode === 'single') {
      singleCard.style.display = 'block';
      temporalCard.style.display = 'none';
      modeSingle.classList.add('active');
      modeTemporal.classList.remove('active');
    } else {
      singleCard.style.display = 'none';
      temporalCard.style.display = 'block';
      modeSingle.classList.remove('active');
      modeTemporal.classList.add('active');
    }
  }

  hideResults() {
    document.getElementById('gpsInfoCard').style.display = 'none';
    document.getElementById('statsCard').style.display = 'none';
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('buildingsCard').style.display = 'none';
    const fieldVisitsCard = document.getElementById('fieldVisitsCard');
    if (fieldVisitsCard) fieldVisitsCard.style.display = 'none';
  }

  handleFile(file, mode) {
    console.log(`🖼️ Handling file for ${mode}:`, file?.name);
    
    if (!this.validateFile(file)) return;

    if (mode === 'single') {
      this.imageFile = file;
    } else if (mode === 't1') {
      this.imageFileT1 = file;
    } else {
      this.imageFileT2 = file;
    }

    console.log(`✅ ${mode} file accepted:`, file.name);
    this.displayFilePreview(file, mode);
  }

  validateFile(file) {
    if (!file) {
      this.showError('Please select a file');
      return false;
    }

    if (!file.type.startsWith('image/')) {
      this.showError('Please upload a valid image file');
      return false;
    }

    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
      this.showError(`File too large. Max: 16MB. Your file: ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
      return false;
    }

    return true;
  }

  displayFilePreview(file, mode) {
    const reader = new FileReader();
    reader.onload = (e) => {
      let previewImage, previewContainer, imageInfo, processBtn;
      
      if (mode === 'single') {
        previewImage = document.getElementById('previewImage');
        previewContainer = document.getElementById('previewContainer');
        imageInfo = document.getElementById('imageInfo');
        processBtn = document.getElementById('processBtn');
      } else {
        previewImage = document.getElementById(`previewImage${mode.toUpperCase()}`);
        previewContainer = document.getElementById(`previewContainer${mode.toUpperCase()}`);
        imageInfo = document.getElementById(`imageInfo${mode.toUpperCase()}`);
        processBtn = document.getElementById('processBtnTemporal');
      }

      if (!previewImage || !previewContainer) {
        console.error('❌ Preview elements not found for mode:', mode);
        return;
      }
      
      previewImage.src = e.target.result;
      previewContainer.style.display = 'block';
      previewContainer.classList.add('show');

      const img = new Image();
      img.onload = () => {
        const sizeKB = (file.size / 1024).toFixed(1);
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        const sizeText = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;

        imageInfo.innerHTML = `
          <div class="info-item">
            <div class="info-label">Filename</div>
            <div class="info-value">${file.name}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Dimensions</div>
            <div class="info-value">${img.width} × ${img.height}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Size</div>
            <div class="info-value">${sizeText}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Type</div>
            <div class="info-value">${file.type.split('/')[1].toUpperCase()}</div>
          </div>
        `;

        // Enable process button
        if (mode === 'single') {
          processBtn.disabled = false;
        } else if (this.imageFileT1 && this.imageFileT2) {
          processBtn.disabled = false;
        }
      };
      
      img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
  }

  showError(message) {
    console.error('❌ Error:', message);
    
    // Create styled error notification
    let errorDiv = document.getElementById('errorNotification');
    if (!errorDiv) {
      errorDiv = document.createElement('div');
      errorDiv.id = 'errorNotification';
      errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        z-index: 10000;
        box-shadow: 0 8px 24px rgba(229, 62, 62, 0.4);
        max-width: 400px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        font-weight: 500;
        animation: slideInRight 0.3s ease;
      `;
      document.body.appendChild(errorDiv);
    }
    
    errorDiv.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 20px;">❌</span>
        <span>${message}</span>
      </div>
    `;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
      errorDiv.style.display = 'none';
    }, 5000);
  }

  showSuccess(message) {
    let successDiv = document.getElementById('successNotification');
    if (!successDiv) {
      successDiv = document.createElement('div');
      successDiv.id = 'successNotification';
      successDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        z-index: 10000;
        box-shadow: 0 8px 24px rgba(56, 161, 105, 0.4);
        max-width: 400px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        font-weight: 500;
        animation: slideInRight 0.3s ease;
      `;
      document.body.appendChild(successDiv);
    }
    
    successDiv.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 20px;">✅</span>
        <span>${message}</span>
      </div>
    `;
    successDiv.style.display = 'block';
    
    setTimeout(() => {
      successDiv.style.display = 'none';
    }, 4000);
  }

  async processImage() {
    console.log('🔍 Starting single image processing...');
    
    if (!this.imageFile) {
      this.showError('Please upload an image first');
      return;
    }

    const progressBar = document.getElementById('progressBarSingle');
    const progressFill = document.getElementById('progressFillSingle');
    const processBtn = document.getElementById('processBtn');

    progressBar.classList.add('show');
    processBtn.disabled = true;

    try {
      this.updateProgress(progressFill, 'Uploading...', 10);

      const formData = new FormData();
      formData.append('file', this.imageFile);

      // Add GPS bounds if provided
      const gpsToggle = document.getElementById('gpsToggle');
      if (gpsToggle && gpsToggle.checked) {
        const north = document.getElementById('northBound').value;
        const south = document.getElementById('southBound').value;
        const east = document.getElementById('eastBound').value;
        const west = document.getElementById('westBound').value;

        if (north && south && east && west) {
          formData.append('north', parseFloat(north));
          formData.append('south', parseFloat(south));
          formData.append('east', parseFloat(east));
          formData.append('west', parseFloat(west));
          console.log('📍 GPS Bounds added:', { north, south, east, west });
        }
      }

      this.updateProgress(progressFill, 'Processing with AI...', 40);

      // FIXED: Use /api/predict for JSON response
      const response = await fetch(`${this.backendUrl}/api/predict`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `Server error (${response.status})`);
      }

      this.updateProgress(progressFill, 'Analyzing results...', 70);

      // Parse JSON response
      const data = await response.json();
      console.log('📊 API Response:', data);

      if (data.success) {
        this.currentSessionId = data.session_id;
        this.currentResults = data;
        
        this.updateProgress(progressFill, 'Displaying results...', 90);
        
        await this.displayResults(data, false);
        
        this.updateProgress(progressFill, 'Complete!', 100);
        this.showSuccess('Detection complete!');
      } else {
        throw new Error(data.error || 'Detection failed');
      }

      setTimeout(() => {
        progressBar.classList.remove('show');
        processBtn.disabled = false;
      }, 1500);

    } catch (error) {
      console.error('❌ Processing error:', error);
      this.showError(`Processing failed: ${error.message}\n\nMake sure backend is running on ${this.backendUrl}`);
      progressBar.classList.remove('show');
      processBtn.disabled = false;
    }
  }

  async processTwoTemporal() {
    console.log('🔄 Starting two-temporal processing...');
    
    if (!this.imageFileT1 || !this.imageFileT2) {
      this.showError('Please upload both before and after images');
      return;
    }

    const progressBar = document.getElementById('progressBarTemporal');
    const progressFill = document.getElementById('progressFillTemporal');
    const processBtn = document.getElementById('processBtnTemporal');

    progressBar.classList.add('show');
    processBtn.disabled = true;

    try {
      this.updateProgress(progressFill, 'Uploading images...', 10);

      const formData = new FormData();
      formData.append('file1', this.imageFileT1);
      formData.append('file2', this.imageFileT2);

      // Add threshold
      const threshold = document.getElementById('thresholdSlider').value;
      formData.append('threshold', parseFloat(threshold));
      console.log('🎯 Change threshold:', threshold + '%');

      // Add GPS bounds if provided
      const gpsToggle = document.getElementById('gpsToggleTemporal');
      if (gpsToggle && gpsToggle.checked) {
        const north = document.getElementById('northBoundTemporal').value;
        const south = document.getElementById('southBoundTemporal').value;
        const east = document.getElementById('eastBoundTemporal').value;
        const west = document.getElementById('westBoundTemporal').value;

        if (north && south && east && west) {
          formData.append('north', parseFloat(north));
          formData.append('south', parseFloat(south));
          formData.append('east', parseFloat(east));
          formData.append('west', parseFloat(west));
          console.log('📍 GPS Bounds added:', { north, south, east, west });
        }
      }

      this.updateProgress(progressFill, 'K-means change detection...', 35);

      // FIXED: Use /api/predict for JSON response
      const response = await fetch(`${this.backendUrl}/api/predict`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `Server error (${response.status})`);
      }

      this.updateProgress(progressFill, 'Region proposal detection...', 60);

      // Parse JSON response
      const data = await response.json();
      console.log('📊 API Response:', data);

      if (data.success) {
        this.currentSessionId = data.session_id;
        this.currentResults = data;
        
        this.updateProgress(progressFill, 'Displaying results...', 85);
        
        await this.displayResults(data, true);
        
        this.updateProgress(progressFill, 'Complete!', 100);
        this.showSuccess('Detection complete! Check results below.');
      } else {
        throw new Error(data.error || 'Detection failed');
      }

      setTimeout(() => {
        progressBar.classList.remove('show');
        processBtn.disabled = false;
      }, 1500);

    } catch (error) {
      console.error('❌ Processing error:', error);
      this.showError(`Processing failed: ${error.message}\n\nMake sure backend is running on ${this.backendUrl}`);
      progressBar.classList.remove('show');
      processBtn.disabled = false;
    }
  }

  updateProgress(element, text, percent) {
    if (element) {
      element.style.width = percent + '%';
      element.textContent = `${text} ${percent}%`;
    }
  }

  async displayResults(data, isTemporal = false) {
    console.log('📊 Displaying results, temporal mode:', isTemporal);
    
    const buildings = data.buildings || [];
    const statistics = data.statistics || {};
    const gpsStats = data.gps_stats || {};

    // Display statistics
    this.displayStatistics(statistics, data, isTemporal);

    // Display GPS info if available
    if (gpsStats && gpsStats.bounds) {
      this.displayGPSInfo(gpsStats.bounds);
    }

    // Display buildings list
    this.displayBuildings(buildings, isTemporal);
    
    // Display report section
    this.displayReportSection();
    
    // Display field visits section for illegal buildings
    const illegalBuildings = buildings.filter(b => b.legal_status === 'ILLEGAL');
    if (illegalBuildings.length > 0) {
      this.displayFieldVisitsSection(illegalBuildings.length);
    }
  }

  displayStatistics(stats, data, isTemporal) {
    const statsCard = document.getElementById('statsCard');
    const statsGrid = document.getElementById('statsGrid');

    statsCard.style.display = 'block';

    const gpsEnabled = data.gps_stats ? '🌍 GPS-Enabled' : '📍 Pixel-Based';
    const modeIndicator = isTemporal ? '📅 Two-Temporal (K-means)' : '📷 Single Image';
    const detectionMethod = stats.detection_method || 'region_proposal';

    let statsHTML = `
      <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
        <div class="stat-value">${gpsEnabled}</div>
        <div class="stat-label" style="color: rgba(255,255,255,0.9);">Coordinate System</div>
      </div>
      <div class="stat-card" style="background: linear-gradient(135deg, #5C6BC0 0%, #512DA8 100%); color: white;">
        <div class="stat-value">${modeIndicator}</div>
        <div class="stat-label" style="color: rgba(255,255,255,0.9);">Detection Mode</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.total_buildings || 0}</div>
        <div class="stat-label">Total Buildings</div>
      </div>
    `;

    if (isTemporal) {
      statsHTML += `
        <div class="stat-card" style="background: linear-gradient(135deg, #FFA726 0%, #FB8C00 100%); color: white;">
          <div class="stat-value">${stats.buildings_under_construction || 0}</div>
          <div class="stat-label" style="color: rgba(255,255,255,0.9);">🔄 Under Construction</div>
        </div>
        <div class="stat-card" style="background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%); color: white;">
          <div class="stat-value">${stats.existing_buildings || 0}</div>
          <div class="stat-label" style="color: rgba(255,255,255,0.9);">✅ Existing</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${(stats.change_percentage || 0).toFixed(1)}%</div>
          <div class="stat-label">Overall Change</div>
        </div>
      `;
    }

    statsHTML += `
      <div class="stat-card stat-legal">
        <div class="stat-value">${stats.legal_constructions || 0}</div>
        <div class="stat-label">✅ Legal Buildings</div>
      </div>
      <div class="stat-card stat-illegal">
        <div class="stat-value">${stats.illegal_constructions || 0}</div>
        <div class="stat-label">❌ Illegal Buildings</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${(stats.compliance_rate || 0).toFixed(1)}%</div>
        <div class="stat-label">Compliance Rate</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${(stats.total_building_area_m2 || 0).toLocaleString()}</div>
        <div class="stat-label">Total Area (m²)</div>
      </div>
      <div class="stat-card stat-illegal">
        <div class="stat-value">${(stats.illegal_building_area_m2 || 0).toLocaleString()}</div>
        <div class="stat-label">Illegal Area (m²)</div>
      </div>
      <div class="stat-card" style="background: linear-gradient(135deg, #26A69A 0%, #00897B 100%); color: white;">
        <div class="stat-value">${detectionMethod}</div>
        <div class="stat-label" style="color: rgba(255,255,255,0.9);">Detection Method</div>
      </div>
    `;

    statsGrid.innerHTML = statsHTML;
  }

  displayGPSInfo(bounds) {
    const gpsCard = document.getElementById('gpsInfoCard');
    if (!gpsCard) return;

    gpsCard.style.display = 'block';
    gpsCard.innerHTML = `
      <h3 style="color: #667eea; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
        <span>🌍</span>
        <span>GPS Coverage Area</span>
      </h3>
      <div class="gps-bounds">
        <div class="gps-item">
          <span class="gps-label">⬆️ North:</span>
          <span class="gps-value">${bounds.north?.toFixed(6) || 'N/A'}°</span>
        </div>
        <div class="gps-item">
          <span class="gps-label">⬇️ South:</span>
          <span class="gps-value">${bounds.south?.toFixed(6) || 'N/A'}°</span>
        </div>
        <div class="gps-item">
          <span class="gps-label">➡️ East:</span>
          <span class="gps-value">${bounds.east?.toFixed(6) || 'N/A'}°</span>
        </div>
        <div class="gps-item">
          <span class="gps-label">⬅️ West:</span>
          <span class="gps-value">${bounds.west?.toFixed(6) || 'N/A'}°</span>
        </div>
      </div>
      <div style="margin-top: 15px; padding: 12px; background: #f0f9ff; border-radius: 8px; border-left: 3px solid #4299e1;">
        <p style="margin: 0; color: #2c5282; font-size: 0.9em;">
          <strong>📍 Coverage:</strong> GPS coordinates enable accurate field verification and legal compliance checking.
        </p>
      </div>
    `;
  }

  displayBuildings(buildings, isTemporal) {
    const buildingsCard = document.getElementById('buildingsCard');
    const buildingsList = document.getElementById('buildingsList');

    if (buildings.length === 0) {
      buildingsCard.style.display = 'none';
      return;
    }

    buildingsCard.style.display = 'block';
    buildingsList.innerHTML = '';

    buildings.forEach((building, index) => {
      const confidence = building.confidence || 0;
      const legalStatus = building.legal_status || 'UNKNOWN';
      const legalReason = building.legal_reason || 'Not checked';
      
      let legalClass = '';
      let legalIcon = '';
      if (legalStatus === 'LEGAL') {
        legalClass = 'status-legal';
        legalIcon = '✅';
      } else if (legalStatus === 'ILLEGAL') {
        legalClass = 'status-illegal';
        legalIcon = '❌';
      } else {
        legalClass = 'status-uncertain';
        legalIcon = '⚠️';
      }

      const gpsLat = building.gps_latitude;
      const gpsLon = building.gps_longitude;
      const hasValidGPS = (
          gpsLat !== null && 
          gpsLat !== undefined && 
          !isNaN(gpsLat) &&
          gpsLon !== null && 
          gpsLon !== undefined && 
          !isNaN(gpsLon)
      );

      const changeInfo = isTemporal && building.change_percentage !== undefined ? 
        `<div class="building-detail-row">
          <strong>🔄 Change:</strong> ${building.change_percentage.toFixed(1)}%
        </div>
        <div class="building-detail-row">
          <strong>Status:</strong> ${building.status || 'N/A'}
        </div>` : '';

      const item = document.createElement('div');
      item.className = `building-item ${legalClass}`;
      
      item.innerHTML = `
        <div class="building-header">
          <span class="building-id">${legalIcon} Building #${index + 1}</span>
          <span class="legal-badge ${legalClass}">${legalStatus}</span>
        </div>
        <div class="building-details">
          <div class="building-detail-row">
            <strong>ID:</strong> ${building.id || 'N/A'}
          </div>
          ${changeInfo}
          <div class="building-detail-row">
            <strong>Legal Status:</strong> ${legalStatus}
          </div>
          <div class="building-detail-row">
            <strong>Reason:</strong> ${legalReason}
          </div>
          ${hasValidGPS ? `
            <div class="building-detail-row" style="background: #ebf8ff; padding: 8px; border-radius: 6px; margin-top: 8px;">
              <strong style="color: #2c5282;">🌍 GPS Coordinates:</strong><br>
              <span style="margin-left: 20px; color: #2d3748;">📍 Lat: ${gpsLat.toFixed(6)}°</span><br>
              <span style="margin-left: 20px; color: #2d3748;">📍 Lon: ${gpsLon.toFixed(6)}°</span>
              ${building.gps_accuracy_estimate_m ? 
                `<br><span style="margin-left: 20px; color: #718096; font-size: 0.85em;">±${building.gps_accuracy_estimate_m.toFixed(1)}m accuracy</span>` : ''}
            </div>
          ` : `<div class="building-detail-row">
            <strong>📍 Location:</strong> Pixel (${building.center_x}, ${building.center_y})
          </div>`}
          <div class="building-detail-row">
            <strong>Area:</strong> ${building.area_pixels?.toLocaleString() || 0} pixels 
            (~${building.area_m2 || Math.round((building.area_pixels || 0) * 0.25)} m²)
          </div>
          <div class="building-detail-row">
            <strong>Detection Confidence:</strong> 
            <span style="color: ${confidence > 80 ? '#38a169' : confidence > 60 ? '#ed8936' : '#e53e3e'}; font-weight: 600;">
              ${confidence.toFixed(0)}%
            </span>
          </div>
          ${building.perimeter_m ? `
            <div class="building-detail-row">
              <strong>Perimeter:</strong> ${building.perimeter_m.toFixed(1)} m
            </div>
          ` : ''}
          ${building.detection_method ? `
            <div class="building-detail-row">
              <strong>Method:</strong> ${building.detection_method}
            </div>
          ` : ''}
        </div>
      `;
      buildingsList.appendChild(item);
    });
  }

  displayReportSection() {
    const statsCard = document.getElementById('statsCard');
    
    const existingReport = document.querySelector('.report-section');
    if (existingReport) {
      existingReport.remove();
    }
    
    const reportSection = document.createElement('div');
    reportSection.className = 'report-section';
    reportSection.style.cssText = `
      margin-top: 30px;
      padding: 24px;
      background: linear-gradient(135deg, rgba(67, 160, 71, 0.1) 0%, rgba(46, 125, 50, 0.1) 100%);
      border-radius: 15px;
      border: 2px solid rgba(67, 160, 71, 0.3);
      animation: slideDown 0.5s ease;
    `;
    
    reportSection.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 250px;">
          <h3 style="color: #2e7d32; margin-bottom: 10px; font-size: 1.3em; display: flex; align-items: center; gap: 10px;">
            <span>📄</span>
            <span>Generate Comprehensive Report</span>
          </h3>
          <p style="color: #4a5568; font-size: 0.95em; margin: 0; line-height: 1.6;">
            Create a detailed PDF report with:
            • Violation details & GPS coordinates
            • Statistical analysis & compliance rates
            • Visual detection results with maps
            • Building inventory with legal status
          </p>
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px;">
          <button id="generateReportBtn" onclick="detector.generateReport()" style="
            background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
            color: white;
            border: none;
            padding: 14px 32px;
            font-size: 1.05em;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(67, 160, 71, 0.4);
            white-space: nowrap;
          " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(67, 160, 71, 0.5)';" 
             onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(67, 160, 71, 0.4)';">
            ⬇️ Generate PDF Report
          </button>
          <button onclick="detector.exportGeoJSON()" style="
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            font-size: 0.95em;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(66, 153, 225, 0.4);
            white-space: nowrap;
          " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(66, 153, 225, 0.5)';" 
             onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(66, 153, 225, 0.4)';">
            🗺️ Export GeoJSON
          </button>
        </div>
      </div>
      <div id="reportStatus" style="margin-top: 15px; display: none;"></div>
    `;
    
    statsCard.parentElement.insertBefore(reportSection, statsCard.nextSibling);
  }

  displayFieldVisitsSection(illegalCount) {
    const buildingsCard = document.getElementById('buildingsCard');
    
    const existingSection = document.querySelector('.field-visits-section');
    if (existingSection) {
      existingSection.remove();
    }
    
    const fieldSection = document.createElement('div');
    fieldSection.className = 'field-visits-section';
    fieldSection.style.cssText = `
      margin-top: 30px;
      padding: 24px;
      background: linear-gradient(135deg, rgba(245, 101, 101, 0.1) 0%, rgba(229, 62, 62, 0.1) 100%);
      border-radius: 15px;
      border: 2px solid rgba(245, 101, 101, 0.3);
      animation: slideDown 0.5s ease;
    `;
    
    fieldSection.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 250px;">
          <h3 style="color: #c53030; margin-bottom: 10px; font-size: 1.3em; display: flex; align-items: center; gap: 10px;">
            <span>🚗</span>
            <span>Field Visit Management</span>
          </h3>
          <p style="color: #4a5568; font-size: 0.95em; margin: 0; line-height: 1.6;">
            ${illegalCount} illegal building(s) detected requiring field inspection.
            Create field visit requests for verification and enforcement.
          </p>
        </div>
        <button onclick="detector.createFieldVisits()" style="
          background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
          color: white;
          border: none;
          padding: 14px 32px;
          font-size: 1.05em;
          font-weight: 600;
          border-radius: 50px;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(245, 101, 101, 0.4);
          white-space: nowrap;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(245, 101, 101, 0.5)';" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(245, 101, 101, 0.4)';">
          📋 Create Field Visits
        </button>
      </div>
      <div id="fieldVisitStatus" style="margin-top: 15px; display: none;"></div>
    `;
    
    buildingsCard.parentElement.insertBefore(fieldSection, buildingsCard.nextSibling);
  }

  async generateReport() {
    const reportBtn = document.getElementById('generateReportBtn');
    const reportStatus = document.getElementById('reportStatus');
    
    if (!this.currentSessionId) {
      this.showError('No active session. Please run detection first.');
      return;
    }
    
    const originalText = reportBtn.textContent;
    reportBtn.disabled = true;
    reportBtn.textContent = '⏳ Generating...';
    reportStatus.style.display = 'block';
    reportStatus.innerHTML = `
      <div style="padding: 15px; background: rgba(255, 193, 7, 0.1); border-radius: 8px; border-left: 4px solid #ffc107;">
        <strong style="color: #f57c00;">⏳ Generating PDF report...</strong>
        <div style="margin-top: 8px; color: #666; font-size: 0.9em;">This may take 10-30 seconds depending on the number of buildings and images.</div>
      </div>
    `;
    
    try {
      console.log(`📄 Requesting report for session: ${this.currentSessionId}`);
      
      const response = await fetch(
        `${this.backendUrl}/api/generate-report/${this.currentSessionId}`
      );
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }
      
      // Download PDF
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `BuildingDetectionReport_${this.currentSessionId.substring(0, 16)}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      reportStatus.innerHTML = `
        <div style="padding: 15px; background: rgba(76, 175, 80, 0.1); border-radius: 8px; border-left: 4px solid #4CAF50;">
          <strong style="color: #2E7D32;">✅ PDF Report Generated Successfully!</strong><br>
          <small style="color: #666; margin-top: 5px; display: block;">The report has been downloaded to your device.</small>
        </div>
      `;
      
      reportBtn.textContent = '✅ Report Generated';
      reportBtn.style.background = 'linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%)';
      
      this.showSuccess('PDF report downloaded successfully!');
      
      setTimeout(() => {
        reportBtn.disabled = false;
        reportBtn.textContent = originalText;
        reportBtn.style.background = 'linear-gradient(135deg, #43A047 0%, #2E7D32 100%)';
      }, 3000);
      
    } catch (error) {
      console.error('❌ Report generation error:', error);
      reportStatus.innerHTML = `
        <div style="padding: 15px; background: rgba(244, 67, 54, 0.1); border-radius: 8px; border-left: 4px solid #E53935;">
          <strong style="color: #D32F2F;">❌ Error: ${error.message}</strong>
          <div style="margin-top: 8px; color: #666; font-size: 0.9em;">Make sure the backend is running and the session is valid.</div>
        </div>
      `;
      reportBtn.disabled = false;
      reportBtn.textContent = originalText;
      this.showError('Failed to generate report: ' + error.message);
    }
  }

  async exportGeoJSON() {
    if (!this.currentSessionId) {
      this.showError('No active session. Please run detection first.');
      return;
    }
    
    try {
      console.log(`🗺️ Exporting GeoJSON for session: ${this.currentSessionId}`);
      
      const response = await fetch(
        `${this.backendUrl}/api/export-geojson/${this.currentSessionId}`
      );
      
      if (!response.ok) {
        throw new Error('GeoJSON export failed');
      }
      
      // Download GeoJSON
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `buildings_${this.currentSessionId.substring(0, 16)}.geojson`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      this.showSuccess('GeoJSON file downloaded successfully!');
      
    } catch (error) {
      console.error('❌ GeoJSON export error:', error);
      this.showError('Failed to export GeoJSON: ' + error.message);
    }
  }

  async createFieldVisits() {
    if (!this.currentSessionId) {
      this.showError('No active session. Please run detection first.');
      return;
    }
    
    const fieldVisitStatus = document.getElementById('fieldVisitStatus');
    fieldVisitStatus.style.display = 'block';
    fieldVisitStatus.innerHTML = `
      <div style="padding: 15px; background: rgba(255, 193, 7, 0.1); border-radius: 8px; border-left: 4px solid #ffc107;">
        <strong style="color: #f57c00;">⏳ Creating field visits...</strong>
      </div>
    `;
    
    try {
      console.log(`🚗 Creating field visits for session: ${this.currentSessionId}`);
      
      const response = await fetch(
        `${this.backendUrl}/api/field-visits/create-from-session/${this.currentSessionId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            inspector_name: 'Field Inspector',
            priority: 'high',
            scheduled_date: new Date().toISOString().split('T')[0]
          })
        }
      );
      
      const data = await response.json();
      
      if (data.success) {
        fieldVisitStatus.innerHTML = `
          <div style="padding: 15px; background: rgba(76, 175, 80, 0.1); border-radius: 8px; border-left: 4px solid #4CAF50;">
            <strong style="color: #2E7D32;">✅ Created ${data.created_count} Field Visit(s)</strong>
            <div style="margin-top: 10px; max-height: 150px; overflow-y: auto;">
              ${data.visit_ids.map((id, idx) => `
                <div style="padding: 6px; background: white; margin-top: 5px; border-radius: 4px; font-size: 0.9em;">
                  📋 Visit #${idx + 1}: <code style="color: #5C6BC0;">${id}</code>
                </div>
              `).join('')}
            </div>
          </div>
        `;
        
        this.showSuccess(`Created ${data.created_count} field visit(s) for illegal buildings`);
      } else {
        throw new Error(data.message || 'Field visit creation failed');
      }
      
    } catch (error) {
      console.error('❌ Field visit creation error:', error);
      fieldVisitStatus.innerHTML = `
        <div style="padding: 15px; background: rgba(244, 67, 54, 0.1); border-radius: 8px; border-left: 4px solid #E53935;">
          <strong style="color: #D32F2F;">❌ Error: ${error.message}</strong>
        </div>
      `;
      this.showError('Failed to create field visits: ' + error.message);
    }
  }
}

// Add slide-in animation
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInRight {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  @keyframes slideDown {
    from {
      transform: translateY(-20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }
  
  .building-detail-row {
    margin: 6px 0;
    padding: 4px 0;
  }
`;
document.head.appendChild(style);

// Global detector instance
let detector;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  console.log('🌐 DOM Content Loaded - Initializing BuildingDetector');
  detector = new BuildingDetector();
  console.log('✅ Detector ready. Backend URL:', detector.backendUrl);
});

// Helper function for logout
function logout() {
  if (confirm('Are you sure you want to logout?')) {
    localStorage.clear();
    sessionStorage.clear();
    window.location.href = 'login.html';
  }
}