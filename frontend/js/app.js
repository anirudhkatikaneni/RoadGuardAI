/**
 * RoadGuardAI - Master Frontend Application
 * Handles Navigation, Real-time Webcam, Video & Image Uploads, Chart.js Visualizations
 */

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

let appState = {
  currentView: "welcome",
  webcamStream: null,
  webcamInterval: null,
  webcamActive: false,
  dashboardChart: null,
  datasetChart: null,
  perfChart: null,
};

function initApp() {
  setupNavigation();
  setupImageDetection();
  setupVideoDetection();
  setupWebcamDetection();
  setupHistoryFilters();
  checkSystemStatus();
  
  // Listen for hash changes
  window.addEventListener("hashchange", handleHashChange);
  if (window.location.hash) {
    navigateTo(window.location.hash.replace("#", ""));
  } else {
    navigateTo("welcome");
  }
}

/* ==========================================================================
   Navigation & Routing
   ========================================================================== */
function setupNavigation() {
  document.querySelectorAll("[data-nav]").forEach((elem) => {
    elem.addEventListener("click", (e) => {
      e.preventDefault();
      const target = elem.getAttribute("data-nav");
      navigateTo(target);
    });
  });
}

function handleHashChange() {
  const hash = window.location.hash.replace("#", "");
  if (hash) navigateTo(hash);
}

function navigateTo(viewId) {
  if (appState.webcamActive && viewId !== "webcam") {
    stopWebcam();
  }

  const validViews = [
    "welcome",
    "dashboard",
    "image",
    "video",
    "webcam",
    "performance",
    "dataset",
    "history",
    "features",
    "about",
  ];

  if (!validViews.includes(viewId)) viewId = "welcome";
  appState.currentView = viewId;

  // Update views
  document.querySelectorAll(".page-view").forEach((view) => {
    view.classList.remove("active");
  });
  const targetView = document.getElementById(`view-${viewId}`);
  if (targetView) targetView.classList.add("active");

  // Update sidebar active link
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.remove("active");
    if (item.getAttribute("data-nav") === viewId) {
      item.classList.add("active");
    }
  });

  // Update Topbar Title
  const titles = {
    welcome: "Welcome to RoadGuardAI",
    dashboard: "Analytics & Monitoring Dashboard",
    image: "Image Pothole Detection",
    video: "Video Pothole Analysis",
    webcam: "Real-time Live Webcam Detection",
    performance: "Model Performance & Academic Metrics",
    dataset: "Dataset Architecture & Splits",
    history: "Detection Audit History",
    features: "System Capabilities & Architecture",
    about: "Project & Research Details",
  };
  const titleEl = document.getElementById("current-page-title");
  if (titleEl) titleEl.innerText = titles[viewId] || "RoadGuardAI";

  // Trigger view data loaders
  if (viewId === "dashboard") loadDashboard();
  if (viewId === "performance") loadPerformance();
  if (viewId === "dataset") loadDataset();
  if (viewId === "history") loadHistory();

  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ==========================================================================
   System Health Check
   ========================================================================== */
async function checkSystemStatus() {
  try {
    const health = await api.getHealth();
    const pill = document.getElementById("system-status-pill");
    const text = document.getElementById("system-status-text");
    if (health.model_loaded) {
      text.innerText = `Model Ready (${health.device})`;
      pill.style.background = "rgba(16, 185, 129, 0.12)";
      pill.style.color = "var(--accent-emerald)";
    } else {
      text.innerText = "Model Initializing / Training";
      pill.style.background = "rgba(245, 158, 11, 0.12)";
      pill.style.color = "var(--accent-amber)";
    }
  } catch (err) {
    const text = document.getElementById("system-status-text");
    if (text) text.innerText = "Backend Connecting...";
  }
}

/* ==========================================================================
   Dashboard View
   ========================================================================== */
async function loadDashboard() {
  try {
    const data = await api.getDashboard();
    document.getElementById("stat-images").innerText = data.total_images_analyzed || 0;
    document.getElementById("stat-videos").innerText = data.total_videos_analyzed || 0;
    document.getElementById("stat-potholes").innerText = data.total_potholes_detected || 0;
    document.getElementById("stat-avg-conf").innerText = data.average_confidence
      ? `${(data.average_confidence * 100).toFixed(1)}%`
      : "N/A";

    // Recent activity list
    const recentEl = document.getElementById("dashboard-recent-activity");
    if (recentEl) {
      if (!data.recent_activity || data.recent_activity.length === 0) {
        recentEl.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--text-muted);">No detections recorded yet. Start by uploading an image or video!</div>`;
      } else {
        recentEl.innerHTML = data.recent_activity
          .map(
            (act) => `
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.85rem 1rem; border-bottom: 1px solid var(--border-subtle);">
              <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--accent-cyan);"></span>
                <div>
                  <div style="font-weight: 600; font-size: 0.9rem;">${act.file_name}</div>
                  <div style="font-size: 0.78rem; color: var(--text-muted);">${act.detection_type.toUpperCase()} • ${act.created_at}</div>
                </div>
              </div>
              <div style="text-align: right;">
                <span class="status-badge" style="background: rgba(6, 182, 212, 0.12); color: var(--accent-cyan); border-color: rgba(6, 182, 212, 0.3);">
                  ${act.pothole_count} Potholes
                </span>
              </div>
            </div>
          `
          )
          .join("");
      }
    }

    renderDashboardChart(data);
  } catch (err) {
    showToast("Error loading dashboard data", "error");
  }
}

function renderDashboardChart(data) {
  const ctx = document.getElementById("dashboardChart");
  if (!ctx) return;

  if (appState.dashboardChart) {
    appState.dashboardChart.destroy();
  }

  const counts = [
    data.total_images_analyzed || 0,
    data.total_videos_analyzed || 0,
    data.total_webcam_sessions || 0,
  ];

  appState.dashboardChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Images", "Videos", "Live Webcam"],
      datasets: [
        {
          data: counts.every((v) => v === 0) ? [1, 1, 1] : counts,
          backgroundColor: ["#06b6d4", "#8b5cf6", "#f59e0b"],
          borderColor: "#070b14",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } },
        },
      },
    },
  });
}

/* ==========================================================================
   Image Detection View
   ========================================================================== */
function setupImageDetection() {
  const dropZone = document.getElementById("img-drop-zone");
  const fileInput = document.getElementById("img-file-input");
  const btnDetect = document.getElementById("btn-detect-image");
  const confSlider = document.getElementById("img-conf-slider");
  const confValue = document.getElementById("img-conf-value");
  let selectedFile = null;

  if (confSlider && confValue) {
    confSlider.addEventListener("input", (e) => {
      confValue.innerText = `${Math.round(e.target.value * 100)}%`;
    });
  }

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        handleImageSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleImageSelect(e.target.files[0]);
      }
    });
  }

  function handleImageSelect(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      const origPreview = document.getElementById("img-original-preview");
      origPreview.src = e.target.result;
      origPreview.style.display = "block";
      document.getElementById("img-placeholder-text").style.display = "none";
      btnDetect.removeAttribute("disabled");
      btnDetect.classList.remove("btn-outline");
      btnDetect.classList.add("btn-primary");
    };
    reader.readAsDataURL(file);
  }

  if (btnDetect) {
    btnDetect.addEventListener("click", async () => {
      if (!selectedFile) return;

      const conf = parseFloat(confSlider.value);
      btnDetect.disabled = true;
      btnDetect.innerHTML = `<div class="spinner" style="width: 18px; height: 18px;"></div> Analyzing Potholes...`;

      try {
        const result = await api.detectImage(selectedFile, conf);
        displayImageResults(result);
        showToast(`Detection complete: ${result.pothole_count} potholes identified!`, "success");
      } catch (err) {
        showToast(err.message || "Detection failed", "error");
      } finally {
        btnDetect.disabled = false;
        btnDetect.innerHTML = `<i class="fas fa-search"></i> Detect Potholes`;
      }
    });
  }
}

function displayImageResults(res) {
  const resultCard = document.getElementById("img-result-card");
  const processedImg = document.getElementById("img-processed-result");
  const countEl = document.getElementById("res-pothole-count");
  const confEl = document.getElementById("res-avg-conf");
  const timeEl = document.getElementById("res-proc-time");
  const btnDownload = document.getElementById("btn-download-img");

  resultCard.style.display = "block";
  processedImg.src = res.output_url;
  countEl.innerText = res.pothole_count;
  confEl.innerText = res.avg_confidence > 0 ? `${(res.avg_confidence * 100).toFixed(1)}%` : "0%";
  timeEl.innerText = `${res.processing_time_ms} ms`;

  if (btnDownload) {
    btnDownload.onclick = () => {
      const a = document.createElement("a");
      a.href = res.output_url;
      a.download = res.output_filename || "roadguard_detected.jpg";
      a.click();
    };
  }

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ==========================================================================
   Video Detection View
   ========================================================================== */
function setupVideoDetection() {
  const dropZone = document.getElementById("vid-drop-zone");
  const fileInput = document.getElementById("vid-file-input");
  const btnDetect = document.getElementById("btn-detect-video");
  const confSlider = document.getElementById("vid-conf-slider");
  const confValue = document.getElementById("vid-conf-value");
  let selectedFile = null;

  if (confSlider && confValue) {
    confSlider.addEventListener("input", (e) => {
      confValue.innerText = `${Math.round(e.target.value * 100)}%`;
    });
  }

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) handleVideoSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) handleVideoSelect(e.target.files[0]);
    });
  }

  function handleVideoSelect(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    const video = document.getElementById("vid-original-preview");
    video.src = url;
    video.style.display = "block";
    document.getElementById("vid-placeholder-text").style.display = "none";
    btnDetect.removeAttribute("disabled");
    btnDetect.classList.remove("btn-outline");
    btnDetect.classList.add("btn-primary");
  }

  if (btnDetect) {
    btnDetect.addEventListener("click", async () => {
      if (!selectedFile) return;

      const conf = parseFloat(confSlider.value);
      btnDetect.disabled = true;
      btnDetect.innerHTML = `<div class="spinner" style="width: 18px; height: 18px;"></div> Processing Frames...`;

      const progressCard = document.getElementById("vid-progress-card");
      if (progressCard) progressCard.style.display = "block";

      try {
        const result = await api.detectVideo(selectedFile, conf);
        displayVideoResults(result);
        showToast("Video analysis complete!", "success");
      } catch (err) {
        showToast(err.message || "Video processing failed", "error");
      } finally {
        btnDetect.disabled = false;
        btnDetect.innerHTML = `<i class="fas fa-play"></i> Start Video Analysis`;
        if (progressCard) progressCard.style.display = "none";
      }
    });
  }
}

function displayVideoResults(res) {
  const resultCard = document.getElementById("vid-result-card");
  const player = document.getElementById("vid-processed-player");
  const countEl = document.getElementById("vid-res-total-potholes");
  const maxEl = document.getElementById("vid-res-max-frame");
  const fpsEl = document.getElementById("vid-res-fps");
  const timeEl = document.getElementById("vid-res-time");
  const btnDownload = document.getElementById("btn-download-vid");

  resultCard.style.display = "block";
  player.src = res.output_url;
  player.load();

  countEl.innerText = res.total_potholes_detected;
  maxEl.innerText = res.max_potholes_in_single_frame;
  fpsEl.innerText = `${res.average_fps} fps`;
  timeEl.innerText = `${res.processing_time_seconds} s`;

  if (btnDownload) {
    btnDownload.onclick = () => {
      const a = document.createElement("a");
      a.href = res.output_url;
      a.download = res.output_filename || "roadguard_video.mp4";
      a.click();
    };
  }

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ==========================================================================
   Real-Time Webcam Detection
   ========================================================================== */
function setupWebcamDetection() {
  const btnStart = document.getElementById("btn-start-webcam");
  const btnStop = document.getElementById("btn-stop-webcam");
  const video = document.getElementById("webcam-stream");
  const outputImg = document.getElementById("webcam-annotated-output");
  const countEl = document.getElementById("webcam-pothole-counter");
  const fpsEl = document.getElementById("webcam-fps-counter");
  const confSlider = document.getElementById("webcam-conf-slider");

  let frameCount = 0;
  let lastTime = performance.now();
  let isSending = false;

  btnStart.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" },
        audio: false,
      });

      video.srcObject = stream;
      video.play();
      appState.webcamStream = stream;
      appState.webcamActive = true;

      btnStart.style.display = "none";
      btnStop.style.display = "inline-flex";
      outputImg.style.display = "block";

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Frame capture loop (~12 FPS)
      appState.webcamInterval = setInterval(async () => {
        if (!appState.webcamActive || isSending || video.readyState !== 4) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const base64Data = canvas.toDataURL("image/jpeg", 0.75);
        const conf = confSlider ? parseFloat(confSlider.value) : 0.25;

        isSending = true;
        try {
          const res = await api.detectFrame(base64Data, conf);
          if (res.annotated_frame_base64) {
            outputImg.src = res.annotated_frame_base64;
          }
          if (countEl) countEl.innerText = res.pothole_count;

          frameCount++;
          const now = performance.now();
          if (now - lastTime >= 1000) {
            const currentFps = (frameCount * 1000) / (now - lastTime);
            if (fpsEl) fpsEl.innerText = `${currentFps.toFixed(1)} FPS`;
            frameCount = 0;
            lastTime = now;
          }
        } catch (err) {
          console.error("Frame detection error:", err);
        } finally {
          isSending = false;
        }
      }, 80);

      showToast("Webcam active. Real-time inference running!", "success");
    } catch (err) {
      showToast(`Camera access failed: ${err.message}`, "error");
    }
  });

  btnStop.addEventListener("click", () => {
    stopWebcam();
    showToast("Webcam stopped.", "info");
  });
}

function stopWebcam() {
  if (appState.webcamInterval) {
    clearInterval(appState.webcamInterval);
    appState.webcamInterval = null;
  }
  if (appState.webcamStream) {
    appState.webcamStream.getTracks().forEach((t) => t.stop());
    appState.webcamStream = null;
  }
  appState.webcamActive = false;

  const btnStart = document.getElementById("btn-start-webcam");
  const btnStop = document.getElementById("btn-stop-webcam");
  const outputImg = document.getElementById("webcam-annotated-output");

  if (btnStart) btnStart.style.display = "inline-flex";
  if (btnStop) btnStop.style.display = "none";
  if (outputImg) outputImg.style.display = "none";
}

/* ==========================================================================
   Model Performance View
   ========================================================================== */
async function loadPerformance() {
  try {
    const data = await api.getModelMetrics();
    const statusData = await api.getModelStatus();

    if (data.status === "not_evaluated") {
      document.getElementById("perf-not-trained-msg").style.display = "block";
      document.getElementById("perf-metrics-content").style.display = "none";
      return;
    }

    document.getElementById("perf-not-trained-msg").style.display = "none";
    document.getElementById("perf-metrics-content").style.display = "block";

    const m = data.metrics || {};
    document.getElementById("metric-precision").innerText = `${(m.precision * 100).toFixed(1)}%`;
    document.getElementById("metric-recall").innerText = `${(m.recall * 100).toFixed(1)}%`;
    document.getElementById("metric-f1").innerText = `${(m.f1_score * 100).toFixed(1)}%`;
    document.getElementById("metric-map50").innerText = `${(m.mAP50 * 100).toFixed(1)}%`;
    document.getElementById("metric-map5095").innerText = `${(m.mAP50_95 * 100).toFixed(1)}%`;

    // Metadata details
    const meta = statusData.training_metadata || {};
    document.getElementById("perf-device").innerText = meta.device_name || "CUDA GPU";
    document.getElementById("perf-epochs").innerText = meta.epochs || 25;
    document.getElementById("perf-duration").innerText = meta.training_duration_human || "N/A";
    document.getElementById("perf-test-images").innerText = data.total_test_images || 218;

    renderPerformanceChart(m);
  } catch (err) {
    console.error("Error loading performance:", err);
  }
}

function renderPerformanceChart(m) {
  const ctx = document.getElementById("perfRadarChart");
  if (!ctx) return;

  if (appState.perfChart) appState.perfChart.destroy();

  appState.perfChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Precision", "Recall", "F1-Score", "mAP@50", "mAP@50-95"],
      datasets: [
        {
          label: "Test Accuracy (%)",
          data: [
            m.precision * 100,
            m.recall * 100,
            m.f1_score * 100,
            m.mAP50 * 100,
            m.mAP50_95 * 100,
          ],
          backgroundColor: [
            "rgba(6, 182, 212, 0.7)",
            "rgba(59, 130, 246, 0.7)",
            "rgba(139, 92, 246, 0.7)",
            "rgba(16, 185, 129, 0.7)",
            "rgba(245, 158, 11, 0.7)",
          ],
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
        x: {
          ticks: { color: "#94a3b8" },
          grid: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

/* ==========================================================================
   Dataset View
   ========================================================================== */
async function loadDataset() {
  try {
    const data = await api.getDatasetSummary();
    if (data.status === "not_ready") return;

    document.getElementById("ds-total-images").innerText = data.total_images.toLocaleString();
    document.getElementById("ds-total-annotations").innerText = data.total_annotations.toLocaleString();
    document.getElementById("ds-train-split").innerText = `${data.splits.train.images} (${data.splits.train.percentage}%)`;
    document.getElementById("ds-val-split").innerText = `${data.splits.val.images} (${data.splits.val.percentage}%)`;
    document.getElementById("ds-test-split").innerText = `${data.splits.test.images} (${data.splits.test.percentage}%)`;
    document.getElementById("ds-unique-scenes").innerText = data.data_quality_report.unique_scenes;

    renderDatasetChart(data.splits);
  } catch (err) {
    console.error("Error loading dataset:", err);
  }
}

function renderDatasetChart(splits) {
  const ctx = document.getElementById("datasetSplitChart");
  if (!ctx) return;

  if (appState.datasetChart) appState.datasetChart.destroy();

  appState.datasetChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Train (70%)", "Validation (20%)", "Test (10%)"],
      datasets: [
        {
          data: [splits.train.images, splits.val.images, splits.test.images],
          backgroundColor: ["#06b6d4", "#3b82f6", "#8b5cf6"],
          borderColor: "#070b14",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8" },
        },
      },
    },
  });
}

/* ==========================================================================
   History View
   ========================================================================== */
function setupHistoryFilters() {
  const filterSelect = document.getElementById("history-type-filter");
  if (filterSelect) {
    filterSelect.addEventListener("change", (e) => {
      loadHistory(e.target.value || null);
    });
  }
}

async function loadHistory(filterType = null) {
  try {
    const res = await api.getHistory(50, 0, filterType);
    const tbody = document.getElementById("history-table-body");
    if (!tbody) return;

    if (!res.records || res.records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2.5rem;">No detection records found.</td></tr>`;
      return;
    }

    tbody.innerHTML = res.records
      .map(
        (rec) => `
        <tr>
          <td><strong style="color: var(--text-primary);">${rec.file_name}</strong></td>
          <td>
            <span class="status-badge" style="background: rgba(6, 182, 212, 0.12); color: var(--accent-cyan); border-color: rgba(6, 182, 212, 0.25);">
              ${rec.detection_type.toUpperCase()}
            </span>
          </td>
          <td><strong style="color: var(--accent-amber);">${rec.pothole_count}</strong></td>
          <td>${rec.avg_confidence > 0 ? `${(rec.avg_confidence * 100).toFixed(1)}%` : "N/A"}</td>
          <td>${rec.created_at}</td>
          <td>
            <div style="display: flex; gap: 0.5rem;">
              <a href="/api/results/${rec.result_path}" target="_blank" class="btn btn-outline btn-sm" title="View Full Result">
                <i class="fas fa-external-link-alt"></i>
              </a>
              <button onclick="deleteRecord(${rec.id})" class="btn btn-outline btn-sm" style="color: var(--accent-rose);" title="Delete Record">
                <i class="fas fa-trash"></i>
              </button>
            </div>
          </td>
        </tr>
      `
      )
      .join("");
  } catch (err) {
    showToast("Error loading history", "error");
  }
}

window.deleteRecord = async function (id) {
  if (!confirm("Are you sure you want to delete this detection record?")) return;
  try {
    await api.deleteHistory(id);
    showToast("Record deleted", "success");
    loadHistory();
  } catch (err) {
    showToast("Failed to delete record", "error");
  }
};

/* ==========================================================================
   Toast Notification Utility
   ========================================================================== */
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  const icon =
    type === "success"
      ? "fa-check-circle"
      : type === "error"
      ? "fa-exclamation-circle"
      : "fa-info-circle";
  toast.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
