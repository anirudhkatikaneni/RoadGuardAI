/**
 * RoadGuardAI - Client API Layer
 */

const API_BASE = "";

const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("Failed to check health");
    return res.json();
  },

  async getDashboard() {
    const res = await fetch(`${API_BASE}/api/dashboard`);
    if (!res.ok) throw new Error("Failed to fetch dashboard");
    return res.json();
  },

  async getModelStatus() {
    const res = await fetch(`${API_BASE}/api/model/status`);
    if (!res.ok) throw new Error("Failed to fetch model status");
    return res.json();
  },

  async getModelMetrics() {
    const res = await fetch(`${API_BASE}/api/model/metrics`);
    if (!res.ok) throw new Error("Failed to fetch model metrics");
    return res.json();
  },

  async getDatasetSummary() {
    const res = await fetch(`${API_BASE}/api/dataset/summary`);
    if (!res.ok) throw new Error("Failed to fetch dataset summary");
    return res.json();
  },

  async detectImage(file, conf = 0.25) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("conf_threshold", conf);

    const res = await fetch(`${API_BASE}/api/detect/image`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Detection failed" }));
      throw new Error(err.detail || "Image detection failed");
    }
    return res.json();
  },

  async detectVideo(file, conf = 0.25) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("conf_threshold", conf);

    const res = await fetch(`${API_BASE}/api/detect/video`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Video detection failed" }));
      throw new Error(err.detail || "Video detection failed");
    }
    return res.json();
  },

  async detectFrame(base64Image, conf = 0.25) {
    const formData = new FormData();
    formData.append("frame_base64", base64Image);
    formData.append("conf_threshold", conf);

    const res = await fetch(`${API_BASE}/api/detect/frame`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Frame detection failed");
    return res.json();
  },

  async getHistory(limit = 50, offset = 0, type = null) {
    let url = `${API_BASE}/api/history?limit=${limit}&offset=${offset}`;
    if (type) url += `&type=${type}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch history");
    return res.json();
  },

  async deleteHistory(id) {
    const res = await fetch(`${API_BASE}/api/history/${id}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Failed to delete record");
    return res.json();
  }
};
