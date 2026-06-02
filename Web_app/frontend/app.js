const MODE_HANDS = "hands";
const MODE_HOLISTIC = "holistic";

const video = document.getElementById("video");
const statusLine = document.getElementById("status");
const predictionLine = document.getElementById("prediction");
const statsLine = document.getElementById("stats");
const analysisLine = document.getElementById("analysis");

const startStreamButton = document.getElementById("startStreamButton");
const stopStreamButton = document.getElementById("stopStreamButton");
const switchCameraButton = document.getElementById("switchCameraButton");
const durationInput = document.getElementById("durationInput");
const modeSelect = document.getElementById("modeSelect");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");

const startControls = document.getElementById("startControls");
const streamControls = document.getElementById("streamControls");

const bufferProgress = document.getElementById("bufferProgress");
const bufferText = document.getElementById("bufferText");
const progressWrap = document.getElementById("progressWrap");
const infoWrap = document.getElementById("infoWrap");
const recognizedText = document.getElementById("recognizedText");
const rawGestureText = document.getElementById("rawGestureText");
const clearTranscriptButton = document.getElementById("clearTranscriptButton");

let mediaStream = null;
let isStreaming = false;
let isStarting = false;
let isPreparing = false;
let stopRequested = false;
let currentFacingMode = "user";

let framesCollected = [];
let captureStartMs = 0;
let lastElapsedSec = 0;
let activeCaptureMode = MODE_HOLISTIC;

let hands = null;
let holistic = null;

function setStatus(text, isError = false) {
  statusLine.textContent = `Статус: ${text}`;
  statusLine.style.color = isError ? "#b3261e" : "#5c6773";
}

function setLoading(text) {
  loadingText.textContent = text;
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

function waitNextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

function updateControls() {
  startStreamButton.disabled = isStreaming || isStarting || isPreparing;
  stopStreamButton.disabled = !isStreaming;
  switchCameraButton.disabled = isStreaming || isStarting || isPreparing;
  durationInput.disabled = isStreaming || isPreparing;
  modeSelect.disabled = isStreaming || isStarting || isPreparing;

  if (isStreaming) {
    startControls.classList.add("hidden");
    streamControls.classList.remove("hidden");
  } else {
    startControls.classList.remove("hidden");
    streamControls.classList.add("hidden");
  }
}

function readDurationSec() {
  const raw = parseFloat(durationInput.value);
  if (Number.isNaN(raw)) {
    return 2.0;
  }
  const clamped = Math.min(Math.max(raw, 0.5), 10);
  durationInput.value = clamped.toString();
  return clamped;
}

function updateProgressTime(elapsedSec, durationSec) {
  bufferProgress.max = durationSec;
  bufferProgress.value = Math.min(elapsedSec, durationSec);
  bufferText.textContent = `${elapsedSec.toFixed(1)}/${durationSec.toFixed(1)}s`;
}

function updateStats(elapsedSec, frames) {
  const safeElapsed = Math.max(elapsedSec, 0.1);
  const fps = frames / safeElapsed;
  statsLine.textContent = `fps: ${fps.toFixed(1)} | кадров: ${frames}`;
}

function setAnalysis(text) {
  analysisLine.textContent = text;
}

function clearTranscript() {
  recognizedText.value = "";
  rawGestureText.value = "";
}

function setTranscript(transcript) {
  recognizedText.value = String(transcript?.final_text || "");
  rawGestureText.value = String(transcript?.raw_text || "");
  recognizedText.scrollTop = recognizedText.scrollHeight;
  rawGestureText.scrollTop = rawGestureText.scrollHeight;
}

function appendRecognizedGesture(label) {
  const cleanLabel = String(label || "").trim();
  if (!cleanLabel || cleanLabel === "no_event") {
    return;
  }
  recognizedText.value = recognizedText.value
    ? `${recognizedText.value} ${cleanLabel}`
    : cleanLabel;
  recognizedText.scrollTop = recognizedText.scrollHeight;
}

function appendRawGesture(label) {
  const cleanLabel = String(label || "").trim();
  if (!cleanLabel || cleanLabel === "no_event") {
    return;
  }
  rawGestureText.value = rawGestureText.value
    ? `${rawGestureText.value} ${cleanLabel}`
    : cleanLabel;
  rawGestureText.scrollTop = rawGestureText.scrollHeight;
}

function safeNum(value) {
  if (value === null || value === undefined) {
    return 0.0;
  }
  const num = Number(value);
  if (Number.isNaN(num) || !Number.isFinite(num)) {
    return 0.0;
  }
  return num;
}

function getSelectedMode() {
  const mode = String(modeSelect.value || MODE_HOLISTIC).toLowerCase();
  return mode === MODE_HOLISTIC ? MODE_HOLISTIC : MODE_HANDS;
}

async function fetchModesConfig() {
  try {
    const response = await fetch("/api/modes");
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const data = await response.json();
    const modes = Array.isArray(data.modes) ? data.modes : [];
    const defaultMode = data.default_mode || MODE_HOLISTIC;

    modeSelect.innerHTML = "";
    for (const item of modes) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.title || item.id;
      modeSelect.appendChild(option);
    }
    if (!modes.length) {
      modeSelect.innerHTML = `
        <option value="hands">Только руки</option>
        <option value="holistic">Руки + поза</option>
      `;
    }
    modeSelect.value = defaultMode;
  } catch (_error) {
    modeSelect.innerHTML = `
      <option value="hands">Только руки</option>
      <option value="holistic">Руки + поза</option>
    `;
    modeSelect.value = MODE_HOLISTIC;
  }
}

function isMobileDevice() {
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const mobileUA = /Android|iPhone|iPad|iPod|Windows Phone/i.test(navigator.userAgent);
  return coarsePointer || mobileUA;
}

async function startCamera() {
  if (mediaStream) {
    return;
  }

  try {
    isStarting = true;
    const mobile = isMobileDevice();
    const videoConstraints = mobile
      ? {
          facingMode: currentFacingMode,
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 24, max: 30 },
        }
      : {
          facingMode: currentFacingMode,
        };

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
    video.srcObject = mediaStream;
    await ensureVideoReady();
  } catch (error) {
    setStatus(`не удалось включить камеру: ${error.message}`, true);
  } finally {
    isStarting = false;
    updateControls();
  }
}

function stopCamera() {
  if (!mediaStream) {
    return;
  }
  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
}

function ensureVideoReady() {
  if (video.readyState >= 2) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const onReady = () => {
      video.removeEventListener("loadeddata", onReady);
      resolve();
    };
    video.addEventListener("loadeddata", onReady);
  });
}

function extractHandsFromHandsResults(results) {
  const points = Array.from({ length: 42 }, () => [0.0, 0.0, 0.0]);
  const allLandmarks = results.multiHandLandmarks || [];
  const allHandedness = results.multiHandedness || [];
  let usedLeft = false;
  let usedRight = false;

  for (let i = 0; i < allLandmarks.length; i += 1) {
    const landmarks = allLandmarks[i] || [];
    const handedness = allHandedness[i] || {};
    const label = String(handedness.label || "").toLowerCase();

    let offset = -1;
    if (label === "left" && !usedLeft) {
      offset = 0;
      usedLeft = true;
    } else if (label === "right" && !usedRight) {
      offset = 21;
      usedRight = true;
    } else if (!usedLeft) {
      offset = 0;
      usedLeft = true;
    } else if (!usedRight) {
      offset = 21;
      usedRight = true;
    }

    if (offset < 0) {
      continue;
    }

    for (let j = 0; j < 21; j += 1) {
      const lm = landmarks[j];
      if (lm) {
        points[offset + j] = [safeNum(lm.x), safeNum(lm.y), safeNum(lm.z)];
      }
    }
  }
  return points;
}

function extractHolisticPoints(results) {
  const points = Array.from({ length: 75 }, () => [0.0, 0.0, 0.0]);
  const left = results.leftHandLandmarks || [];
  for (let i = 0; i < 21; i += 1) {
    const lm = left[i];
    if (lm) {
      points[i] = [safeNum(lm.x), safeNum(lm.y), safeNum(lm.z)];
    }
  }

  const right = results.rightHandLandmarks || [];
  for (let i = 0; i < 21; i += 1) {
    const lm = right[i];
    if (lm) {
      points[21 + i] = [safeNum(lm.x), safeNum(lm.y), safeNum(lm.z)];
    }
  }

  const pose = results.poseLandmarks || [];
  for (let i = 0; i < 33; i += 1) {
    const lm = pose[i];
    if (lm) {
      points[42 + i] = [safeNum(lm.x), safeNum(lm.y), safeNum(lm.z)];
    }
  }
  return points;
}

function initHands() {
  if (hands) {
    return;
  }
  if (typeof Hands === "undefined") {
    throw new Error("MediaPipe Hands не загрузился (нет доступа к CDN)");
  }

  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  const complexity = 0;
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: complexity,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results) => {
    if (!isStreaming || activeCaptureMode !== MODE_HANDS) {
      return;
    }
    framesCollected.push(extractHandsFromHandsResults(results));
  });
}

function initHolistic() {
  if (holistic) {
    return;
  }
  if (typeof Holistic === "undefined") {
    throw new Error("MediaPipe Holistic не загрузился (нет доступа к CDN)");
  }

  holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });
  const complexity = 0;
  holistic.setOptions({
    modelComplexity: complexity,
    smoothLandmarks: true,
    enableSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  holistic.onResults((results) => {
    if (!isStreaming || activeCaptureMode !== MODE_HOLISTIC) {
      return;
    }
    framesCollected.push(extractHolisticPoints(results));
  });
}

async function startSession() {
  const response = await fetch("/api/session/start", { method: "POST" });
  if (!response.ok) {
    throw new Error(`ошибка сервера: ${response.status}`);
  }
}

async function prepareInference(mode) {
  let sendFrame = null;

  setLoading("Инициализация MediaPipe...");
  await waitNextPaint();
  if (mode === MODE_HOLISTIC) {
    initHolistic();
    sendFrame = async () => holistic.send({ image: video });
  } else {
    initHands();
    sendFrame = async () => hands.send({ image: video });
  }

  setLoading("Подключение к модели...");
  await waitNextPaint();
  await startSession();
  setLoading("Разогрев модели...");
  await waitNextPaint();
  await sendFrame();

  hideLoading();
  return sendFrame;
}

async function finishSession(frames, durationSec, mode, segmentIndex) {
  if (!frames.length) {
    setStatus("нет кадров для анализа", true);
    return null;
  }

  try {
    const response = await fetch("/api/session/finish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, mode, duration_sec: durationSec }),
    });
    if (!response.ok) {
      throw new Error(`ошибка сервера: ${response.status}`);
    }
    const data = await response.json();

    if (data.prediction && data.prediction.label && data.prediction.label !== "no_event") {
      const conf = (data.prediction.confidence * 100).toFixed(1);
      predictionLine.textContent = `Распознавание: ${data.prediction.label} (${conf}%)`;
      if (data.transcript) {
        setTranscript(data.transcript);
      } else {
        appendRecognizedGesture(data.prediction.label);
        appendRawGesture(data.prediction.label);
      }
    } else {
      predictionLine.textContent = "Распознавание: —";
      if (data.transcript) {
        setTranscript(data.transcript);
      }
    }

    setAnalysis(
      `Режим: ${mode} | Жест: ${segmentIndex} | Кадров: ${data.frames_collected} -> ${data.resampled_to} (${data.resample_method}-resample)`
    );
    const fpsValue = data.fps ? data.fps.toFixed(1) : (frames.length / durationSec).toFixed(1);
    statsLine.textContent = `fps: ${fpsValue} | кадров: ${data.frames_collected}`;
    setStatus(
      stopRequested
        ? "последний жест обработан, автоматическое распознавание остановлено"
        : "жест распознан, автоматическое распознавание продолжается"
    );
    return data;
  } catch (error) {
    setStatus(`ошибка завершения: ${error.message}`, true);
    return null;
  }
}

async function flushPendingTranscript() {
  try {
    const response = await fetch("/api/session/flush", { method: "POST" });
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    if (data.transcript) {
      setTranscript(data.transcript);
    }
  } catch (_error) {
  }
}

async function captureForDuration(durationSec, mode, sendFrame, segmentIndex) {
  framesCollected = [];
  lastElapsedSec = 0;
  captureStartMs = performance.now();
  activeCaptureMode = mode;

  progressWrap.classList.remove("hidden");
  infoWrap.classList.remove("hidden");
  predictionLine.textContent = "Распознавание: —";
  setAnalysis(`Подготовка жеста ${segmentIndex}...`);
  updateProgressTime(0, durationSec);
  updateStats(0, 0);

  setStatus(`идёт сбор кадров жеста ${segmentIndex} (${mode})`);

  const frameIntervalMs = Math.round(1000 / 30);

  while (isStreaming && !stopRequested) {
    const now = performance.now();
    const elapsedSec = (now - captureStartMs) / 1000.0;
    lastElapsedSec = elapsedSec;
    if (elapsedSec >= durationSec) {
      break;
    }

    const frameStart = performance.now();
    await sendFrame();
    updateProgressTime(elapsedSec, durationSec);
    updateStats(elapsedSec, framesCollected.length);

    const frameElapsed = performance.now() - frameStart;
    const sleepMs = Math.max(0, frameIntervalMs - frameElapsed);
    await new Promise((resolve) => setTimeout(resolve, sleepMs));
  }

  updateProgressTime(durationSec, durationSec);
  const effectiveDuration = Math.max(lastElapsedSec, 0.1);
  return {
    frames: framesCollected,
    durationSec: effectiveDuration,
  };
}

async function runAutoRecognition(durationSec, mode, sendFrame) {
  let segmentIndex = 0;

  while (isStreaming) {
    segmentIndex += 1;
    const captured = await captureForDuration(durationSec, mode, sendFrame, segmentIndex);

    if (!captured.frames.length) {
      if (stopRequested) {
        break;
      }
      setStatus("не удалось собрать кадры для жеста", true);
      break;
    }

    await finishSession(captured.frames, captured.durationSec, mode, segmentIndex);

    if (stopRequested) {
      break;
    }
  }

  await flushPendingTranscript();
  isStreaming = false;
  updateControls();
  setStatus("автоматическое распознавание остановлено");
}

startStreamButton.addEventListener("click", async () => {
  if (isStreaming) {
    return;
  }

  let sendFrame = null;
  await startCamera();
  if (!mediaStream) {
    return;
  }

  const durationSec = readDurationSec();
  const mode = getSelectedMode();

  try {
    isPreparing = true;
    updateControls();
    sendFrame = await prepareInference(mode);

    isPreparing = false;
    isStreaming = true;
    stopRequested = false;
    clearTranscript();
    updateControls();
    await runAutoRecognition(durationSec, mode, sendFrame);
  } catch (error) {
    setStatus(`ошибка запуска: ${error.message}`, true);
    hideLoading();
    isPreparing = false;
    isStreaming = false;
    updateControls();
  }
});

stopStreamButton.addEventListener("click", () => {
  if (!isStreaming) {
    return;
  }
  stopRequested = true;
  setStatus("остановлено пользователем, завершаем");
});

clearTranscriptButton.addEventListener("click", async () => {
  clearTranscript();
  try {
    const response = await fetch("/api/session/clear", { method: "POST" });
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    if (data.transcript) {
      setTranscript(data.transcript);
    }
  } catch (_error) {
  }
});

switchCameraButton.addEventListener("click", async () => {
  if (isStreaming || isStarting) {
    return;
  }
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  stopCamera();
  await startCamera();
});

(async function initApp() {
  await fetchModesConfig();
  updateControls();
  updateProgressTime(0, readDurationSec());
  clearTranscript();
})();
