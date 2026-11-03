(() => {
  "use strict";

  const API_BASE = window.API_BASE || "";
  const DEFAULT_MAX_BYTES = 8 * 1024 * 1024;

  const ICONS = {
    recyclable: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M7.5 8.5 10 4l2.5 4.5"/>
        <path d="M4.6 14.8 2.1 10.3l5.4.1"/>
        <path d="M19.4 14.8l2.5-4.5-4.5-2.6"/>
        <path d="M9.5 20l-2.5-4.5h5"/>
        <path d="M14.5 20l2.5-4.5h5"/>
      </svg>`,
    kitchen: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M20 4C11 4 5 8 5 15a4.5 4.5 0 0 0 4.5 4.5C16.5 19.5 20 12 20 4Z"/>
        <path d="M5 19c2-5 6-8 11-10"/>
      </svg>`,
    hazardous: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="8" width="18" height="11" rx="2.5"/>
        <path d="M7 8V6h10v2M7 12h4"/>
        <path d="M14.5 11.5 12 14.7h3l-2.5 3.2"/>
      </svg>`,
    other: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M4 7h16M9 7V4h6v3"/>
        <path d="M6.5 7 7.5 20h9l1-13"/>
        <path d="M10 11v5M14 11v5"/>
      </svg>`,
  };

  const COLORS = {
    recyclable: "#2563eb",
    kitchen: "#16a34a",
    hazardous: "#dc2626",
    other: "#64748b",
  };

  const FALLBACK_CATEGORIES = [
    {
      code: "recyclable",
      name: "可回收物",
      en_name: "Recyclable",
      bin_name: "蓝色可回收物桶",
      bin_color: COLORS.recyclable,
      summary: "适宜回收和资源化利用的垃圾。",
      items: ["废纸箱", "塑料瓶", "易拉罐", "玻璃瓶"],
    },
    {
      code: "kitchen",
      name: "厨余垃圾",
      en_name: "Food Waste",
      bin_name: "绿色厨余垃圾桶",
      bin_color: COLORS.kitchen,
      summary: "易腐烂的生物质生活废弃物。",
      items: ["剩饭剩菜", "果皮果核", "蛋壳", "茶叶渣"],
    },
    {
      code: "hazardous",
      name: "有害垃圾",
      en_name: "Hazardous",
      bin_name: "红色有害垃圾桶",
      bin_color: COLORS.hazardous,
      summary: "对健康或环境有直接或潜在危害的废弃物。",
      items: ["废电池", "过期药品", "废灯管", "油漆桶"],
    },
    {
      code: "other",
      name: "其他垃圾",
      en_name: "Residual Waste",
      bin_name: "灰色其他垃圾桶",
      bin_color: COLORS.other,
      summary: "除前三类之外的其他生活废弃物。",
      items: ["纸巾", "烟头", "陶瓷碎片", "口罩"],
    },
  ];

  const $ = (selector) => document.querySelector(selector);

  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const dropIdle = $("#dropIdle");
  const previewWrap = $("#previewWrap");
  const previewImage = $("#previewImage");
  const fileMeta = $("#fileMeta");
  const removeBtn = $("#removeBtn");
  const predictBtn = $("#predictBtn");
  const resetBtn = $("#resetBtn");
  const retryBtn = $("#retryBtn");
  const resultPanel = $("#resultPanel");
  const panelStatus = $("#panelStatus");
  const uploadLimit = $("#uploadLimit");
  const resultBody = $("#resultBody");
  const resultLabel = $("#resultLabel");
  const resultSummary = $("#resultSummary");
  const resultIcon = $("#resultIcon");
  const categoryBadge = $("#categoryBadge");
  const confidenceValue = $("#confidenceValue");
  const topBars = $("#topBars");
  const adviceTips = $("#adviceTips");
  const adviceItems = $("#adviceItems");
  const binName = $("#binName");
  const binCampus = $("#binCampus");
  const demoNote = $("#demoNote");
  const errorMessage = $("#errorMessage");
  const guideGrid = $("#guideGrid");
  const statusPill = $("#statusPill");
  const statusText = $("#statusText");

  const state = {
    file: null,
    objectUrl: null,
    busy: false,
  };

  let maxBytes = DEFAULT_MAX_BYTES;

  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ESCAPES[char]);
  const safeColor = (value, fallback) =>
    /^#[0-9a-f]{6}$/i.test(String(value || "")) ? value : fallback;

  const formatBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  function setPanelState(name) {
    resultPanel.dataset.state = name;
  }

  function setBusy(busy) {
    state.busy = busy;
    predictBtn.disabled = busy || !state.file;
    resetBtn.disabled = busy || !state.file;
    removeBtn.disabled = busy;
    predictBtn.classList.toggle("is-loading", busy);
    predictBtn.setAttribute("aria-busy", String(busy));
    predictBtn.querySelector(".btn-label").textContent = busy ? "识别中" : "开始识别";
    dropzone.style.pointerEvents = busy ? "none" : "";
  }

  function revokeObjectUrl() {
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
  }

  function clearFile() {
    revokeObjectUrl();
    state.file = null;
    fileInput.value = "";
    previewImage.removeAttribute("src");
    previewWrap.classList.add("hidden");
    dropIdle.classList.remove("hidden");
    fileMeta.textContent = "";
    predictBtn.disabled = true;
    resetBtn.disabled = true;
    removeBtn.disabled = false;
    resultPanel.style.removeProperty("--accent");
    demoNote.classList.add("hidden");
    panelStatus.textContent = "等待上传";
    setPanelState("empty");
  }

  function showError(message) {
    panelStatus.textContent = "识别失败";
    errorMessage.textContent = message;
    setPanelState("error");
  }

  function selectFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      clearFile();
      showError("请选择图片文件（JPG、PNG、WEBP 或 BMP）。");
      return;
    }
    if (file.size > maxBytes) {
      clearFile();
      showError(`图片大小不能超过 ${formatBytes(maxBytes)}，请压缩后重试。`);
      return;
    }

    revokeObjectUrl();
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);
    previewImage.src = state.objectUrl;
    previewWrap.classList.remove("hidden");
    dropIdle.classList.add("hidden");

    const name = file.name || "camera.jpg";
    fileMeta.textContent = `${name.length > 34 ? `${name.slice(0, 31)}...` : name} · ${formatBytes(file.size)}`;

    predictBtn.disabled = false;
    resetBtn.disabled = false;
    removeBtn.disabled = false;
    resultPanel.style.removeProperty("--accent");
    demoNote.classList.add("hidden");
    panelStatus.textContent = "图片已就绪";
    setPanelState("empty");
  }

  function renderBars(items, topCategory) {
    topBars.innerHTML = items
      .map((item) => {
        const percent = Math.max(0, Math.min(100, Number(item.probability) * 100));
        const color = safeColor(COLORS[item.category], "#158a57");
        const isTop = item.category === topCategory;
        return `
          <div class="bar-row${isTop ? " is-top" : ""}">
            <span>${esc(item.label)}</span>
            <div class="bar-track">
              <div class="bar-fill" style="--bar-color:${color}" data-width="${percent.toFixed(1)}"></div>
            </div>
            <b>${percent.toFixed(1)}%</b>
          </div>`;
      })
      .join("");

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        topBars.querySelectorAll(".bar-fill").forEach((bar) => {
          bar.style.width = `${bar.dataset.width}%`;
        });
      });
    });
  }

  function renderResult(data) {
    const advice = data.advice || {};
    const accent = safeColor(advice.bin_color || COLORS[data.category], "#158a57");
    resultPanel.style.setProperty("--accent", accent);

    const confidence = Math.max(0, Math.min(1, Number(data.confidence) || 0));
    categoryBadge.textContent = data.label || advice.name || "未知类别";
    resultLabel.textContent = data.label || advice.name || "未知类别";
    resultSummary.textContent = advice.summary || "";
    confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
    resultIcon.innerHTML = ICONS[data.category] || ICONS.other;
    binName.textContent = advice.bin_name || "对应分类垃圾桶";
    binCampus.textContent = advice.campus || "";

    renderBars(Array.isArray(data.top_k) ? data.top_k : [], data.category);

    adviceTips.innerHTML = (advice.tips || []).map((tip) => `<li>${esc(tip)}</li>`).join("");
    adviceItems.innerHTML = (advice.items || [])
      .map((item) => `<span>${esc(item)}</span>`)
      .join("");

    if (data.demo_mode) {
      demoNote.textContent = data.demo_note || "当前为演示模式结果，加载训练权重后将输出真实模型推理。";
      demoNote.classList.remove("hidden");
    } else {
      demoNote.classList.add("hidden");
    }

    setPanelState("result");

    if (window.matchMedia("(max-width: 1020px)").matches) {
      resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  async function predict() {
    if (!state.file || state.busy) return;

    setBusy(true);
    setPanelState("loading");
    panelStatus.textContent = "模型推理中";

    try {
      const form = new FormData();
      form.append("file", state.file, state.file.name || "camera.jpg");

      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: form,
      });

      let payload;
      try {
        payload = await response.json();
      } catch (parseError) {
        throw new Error(`服务返回异常（HTTP ${response.status}），请确认后端已启动。`);
      }

      if (!response.ok || !payload.success) {
        throw new Error(payload?.error?.message || `请求失败（HTTP ${response.status}）。`);
      }

      const latency = payload.data?.latency_ms;
      panelStatus.textContent = latency != null ? `完成 · ${latency} ms` : "识别完成";
      renderResult(payload.data);
    } catch (error) {
      showError(error.message || "网络异常，请检查后端服务是否启动。");
    } finally {
      setBusy(false);
    }
  }

  function renderGuide(categories) {
    guideGrid.innerHTML = categories
      .map((item) => {
        const accent = safeColor(item.bin_color || COLORS[item.code], "#158a57");
        const items = (item.items || [])
          .slice(0, 4)
          .map((name) => `<span>${esc(name)}</span>`)
          .join("");
        return `
          <article class="guide-card" style="--accent:${accent}">
            <div class="guide-icon">${ICONS[item.code] || ICONS.other}</div>
            <h3>${esc(item.name)}</h3>
            <p class="en">${esc(item.en_name || "")}</p>
            <p>${esc(item.summary || "")}</p>
            <div class="guide-items">${items}</div>
            <div class="guide-bin"><i></i>${esc(item.bin_name || "")}</div>
          </article>`;
      })
      .join("");
  }

  async function loadGuide() {
    try {
      const response = await fetch(`${API_BASE}/api/categories`);
      const payload = await response.json();
      if (!response.ok || !payload.success || !Array.isArray(payload.data)) {
        throw new Error("catalog unavailable");
      }
      renderGuide(payload.data);
    } catch (error) {
      renderGuide(FALLBACK_CATEGORIES);
    }
  }

  async function checkHealth() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);

    try {
      const response = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
      const payload = await response.json();
      if (!response.ok || !payload.success) throw new Error("health check failed");

      const data = payload.data;
      maxBytes = Number(data.max_upload_mb) > 0 ? Number(data.max_upload_mb) * 1024 * 1024 : DEFAULT_MAX_BYTES;
      if (uploadLimit) {
        uploadLimit.textContent = `JPG · PNG · WEBP · BMP · 最大 ${formatBytes(maxBytes)}`;
      }

      statusPill.classList.remove("is-ready", "is-demo", "is-offline");
      if (data.ready) {
        statusPill.classList.add("is-ready");
        statusText.textContent = "模型就绪";
        statusPill.title = `推理模型：${data.model?.arch || "CNN"}`;
      } else if (data.demo_mode) {
        statusPill.classList.add("is-demo");
        statusText.textContent = "演示模式";
        statusPill.title = data.load_error || "未加载训练权重，当前使用演示模式";
      } else {
        statusPill.classList.add("is-offline");
        statusText.textContent = "模型未就绪";
        statusPill.title = data.load_error || "请先训练模型并放入 weights/best.pt";
      }
    } catch (error) {
      statusPill.classList.remove("is-ready", "is-demo");
      statusPill.classList.add("is-offline");
      statusText.textContent = "服务离线";
      statusPill.title = "无法连接 Flask 后端，请运行 python app.py";
    } finally {
      clearTimeout(timer);
    }
  }

  // ------------------------------------------------------------------ 事件
  dropzone.addEventListener("click", (event) => {
    if (state.busy || event.target.closest(".remove-btn")) return;
    fileInput.click();
  });

  dropzone.addEventListener("keydown", (event) => {
    if (state.busy) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!state.busy) dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    if (state.busy) return;
    const file = event.dataTransfer?.files?.[0];
    if (file) selectFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) selectFile(fileInput.files[0]);
  });

  removeBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!state.busy) clearFile();
  });

  predictBtn.addEventListener("click", predict);
  retryBtn.addEventListener("click", predict);
  resetBtn.addEventListener("click", () => {
    if (!state.busy) clearFile();
  });

  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  window.addEventListener("beforeunload", revokeObjectUrl);

  // ------------------------------------------------------------------ 初始化
  checkHealth();
  loadGuide();
})();
