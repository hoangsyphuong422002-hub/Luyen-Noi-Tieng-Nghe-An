// VASA Frontend Logic (Web Audio WAV Recorder + Chart.js F0 Visualizer)
// Supports both standalone monolith or Vercel static frontend with remote API
const API_BASE = window.VASA_API_BASE || "";

let challenges = [];
let currentChallengeIdx = 0;
let f0Chart = null;

// Audio Recording State
let audioCtx = null;
let mediaStream = null;
let scriptProcessor = null;
let mediaSource = null;
let recordedBuffers = [];
let recordingSampleRate = 16000;
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;
let holdMode = false;

// DOM Elements
const challengeTitleEl = document.getElementById("challengeTitle");
const challengeTextEl = document.getElementById("challengeText");
const challengeMeaningEl = document.getElementById("challengeMeaning");
const challengeTipEl = document.getElementById("challengeTip");
const nextChallengeBtn = document.getElementById("nextChallengeBtn");
const playSampleBtn = document.getElementById("playSampleBtn");
const testNativeBtn = document.getElementById("testNativeBtn");
const sampleAudioPlayer = document.getElementById("sampleAudioPlayer");

const micVisualizer = document.getElementById("micVisualizer");
const recordBtn = document.getElementById("recordBtn");
const micIcon = document.getElementById("micIcon");
const recordLabel = document.getElementById("recordLabel");
const recordStatus = document.getElementById("recordStatus");
const timerDisplay = document.getElementById("timerDisplay");

const userPlaybackWrap = document.getElementById("userPlaybackWrap");
const userAudioPlayer = document.getElementById("userAudioPlayer");
const uploadWavInput = document.getElementById("uploadWavInput");

const scoreRingFill = document.getElementById("scoreRingFill");
const scoreValueEl = document.getElementById("scoreValue");
const rankBadgeEl = document.getElementById("rankBadge");
const feedbackTextEl = document.getElementById("feedbackText");
const mseValueEl = document.getElementById("mseValue");

// 1. Initialize Chart.js
function initChart() {
    const ctx = document.getElementById("f0Chart").getContext("2d");
    const labels = Array.from({ length: 200 }, (_, i) => i + 1);
    const zeroData = Array(200).fill(0);

    f0Chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Giọng của bạn (Original F0)",
                    data: [...zeroData],
                    borderColor: "#A94F3D",
                    backgroundColor: "rgba(169, 79, 61, 0.12)",
                    borderWidth: 2.8,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "#A94F3D",
                    tension: 0.35,
                    fill: false
                },
                {
                    label: "Chuẩn ngữ điệu Nghệ An (AI Reconstructed F0)",
                    data: [...zeroData],
                    borderColor: "#344E41",
                    backgroundColor: "rgba(120, 150, 107, 0.18)",
                    borderWidth: 2.6,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "#344E41",
                    tension: 0.35,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: "rgba(52, 78, 65, 0.94)",
                    titleColor: "#E8C77B",
                    bodyColor: "#FBF8F1",
                    borderColor: "rgba(232, 199, 123, 0.4)",
                    borderWidth: 1,
                    callbacks: {
                        title: (items) => `Điểm thời gian: ${items[0].label} / 200`,
                        label: (item) => `${item.dataset.label}: ${Number(item.raw).toFixed(2)} semitones`
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: "rgba(155, 114, 83, 0.10)"
                    },
                    ticks: {
                        color: "#5C5346",
                        maxTicksLimit: 10
                    },
                    title: {
                        display: true,
                        text: "Tiến trình câu nói (200 điểm chuẩn hóa thời gian)",
                        color: "#5C5346",
                        font: { size: 11, weight: "600" }
                    }
                },
                y: {
                    grid: {
                        color: "rgba(155, 114, 83, 0.14)"
                    },
                    ticks: {
                        color: "#5C5346"
                    },
                    title: {
                        display: true,
                        text: "Cao độ tương đối (Semitones)",
                        color: "#5C5346",
                        font: { size: 11, weight: "600" }
                    },
                    suggestedMin: -8,
                    suggestedMax: 8
                }
            }
        }
    });
}

// 2. Load Challenges from Backend
async function loadChallenges() {
    try {
        const res = await fetch(`${API_BASE}/api/challenges`);
        if (!res.ok) throw new Error("Failed to fetch challenges");
        const data = await res.json();
        if (data.challenges && data.challenges.length > 0) {
            challenges = data.challenges;
            renderChallenge(0);
        }
    } catch (err) {
        console.warn("Could not load remote challenges, using default:", err);
    }
}

function renderChallenge(idx) {
    if (!challenges.length) return;
    currentChallengeIdx = (idx + challenges.length) % challenges.length;
    const item = challenges[currentChallengeIdx];

    challengeTitleEl.textContent = item.title;
    challengeTextEl.textContent = `"${item.dialect_text}"`;
    challengeMeaningEl.textContent = `💡 Thông tin: ${item.meaning}`;
    challengeTipEl.textContent = `🎯 Mẹo: ${item.tip}`;

    if (item.sample_audio) {
        sampleAudioPlayer.src = item.sample_audio.startsWith("http") ? item.sample_audio : `${API_BASE}${item.sample_audio}`;
        playSampleBtn.disabled = false;
        testNativeBtn.disabled = false;
    } else {
        playSampleBtn.disabled = true;
        testNativeBtn.disabled = true;
    }
}

nextChallengeBtn.addEventListener("click", () => {
    renderChallenge(currentChallengeIdx + 1);
});

playSampleBtn.addEventListener("click", () => {
    if (!sampleAudioPlayer.src) return;
    if (sampleAudioPlayer.paused) {
        sampleAudioPlayer.play();
        playSampleBtn.textContent = "⏸️ Đang phát mẫu...";
    } else {
        sampleAudioPlayer.pause();
        sampleAudioPlayer.currentTime = 0;
        playSampleBtn.textContent = "🔊 Nghe Giọng Mẫu Bản Địa";
    }
});

sampleAudioPlayer.addEventListener("ended", () => {
    playSampleBtn.textContent = "🔊 Nghe Giọng Mẫu Bản Địa";
});

testNativeBtn.addEventListener("click", async () => {
    const item = challenges[currentChallengeIdx];
    const audioUrl = item ? item.sample_audio : "/dataset_audios/37_0348.wav";
    const filename = audioUrl.split("/").pop();

    setLoadingState("Đang chấm điểm file mẫu bản địa...");
    try {
        const res = await fetch(`${API_BASE}/api/score-sample/${encodeURIComponent(filename)}`, {
            method: "POST"
        });
        if (!res.ok) throw new Error("Lỗi khi chấm file mẫu");
        const result = await res.json();
        displayResult(result);
        recordStatus.textContent = `Đã chấm mẫu giọng bản địa (${filename})!`;
    } catch (err) {
        recordStatus.textContent = "Lỗi: " + err.message;
    }
});

// 3. Web Audio API Recorder -> Standard 16-bit PCM WAV Blob
async function startRecording() {
    if (isRecording) return;
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: false
            }
        });

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        recordingSampleRate = audioCtx.sampleRate;
        mediaSource = audioCtx.createMediaStreamSource(mediaStream);

        // Buffer size 4096, 1 input channel, 1 output channel
        scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
        recordedBuffers = [];

        scriptProcessor.onaudioprocess = (event) => {
            if (!isRecording) return;
            const inputData = event.inputBuffer.getChannelData(0);
            recordedBuffers.push(new Float32Array(inputData));
        };

        mediaSource.connect(scriptProcessor);
        scriptProcessor.connect(audioCtx.destination);

        isRecording = true;
        recordStartTime = performance.now();
        updateRecordingUI(true);

        timerInterval = setInterval(() => {
            const elapsed = (performance.now() - recordStartTime) / 1000;
            const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const secs = (elapsed % 60).toFixed(1).padStart(4, "0");
            timerDisplay.textContent = `${mins}:${secs}`;
        }, 100);
    } catch (err) {
        alert("Không thể truy cập Microphone! Vui lòng cấp quyền thu âm hoặc dùng tính năng tải file .wav lên.\nChi tiết: " + err.message);
    }
}

async function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    clearInterval(timerInterval);
    updateRecordingUI(false);

    const durationSec = (performance.now() - recordStartTime) / 1000;

    // Stop tracks and nodes
    if (scriptProcessor) scriptProcessor.disconnect();
    if (mediaSource) mediaSource.disconnect();
    if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
    }
    if (audioCtx && audioCtx.state !== "closed") {
        await audioCtx.close();
    }

    if (durationSec < 0.6 || recordedBuffers.length === 0) {
        recordStatus.textContent = "Đoạn thu âm quá ngắn (< 0.6 giây). Hãy giữ nút lâu hơn và nói trọn câu nhé!";
        return;
    }

    // Encode PCM buffers into a standard .wav Blob
    const wavBlob = encodeWAV(recordedBuffers, recordingSampleRate);
    const wavUrl = URL.createObjectURL(wavBlob);
    userAudioPlayer.src = wavUrl;
    userPlaybackWrap.classList.remove("hidden");

    await uploadAndScoreWav(wavBlob, "recording.wav");
}

function encodeWAV(buffers, sampleRate) {
    let totalLength = 0;
    for (const b of buffers) {
        totalLength += b.length;
    }
    const samples = new Float32Array(totalLength);
    let offset = 0;
    for (const b of buffers) {
        samples.set(b, offset);
        offset += b.length;
    }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // RIFF identifier
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true);  // PCM format = 1
    view.setUint16(22, 1, true);  // Mono = 1 channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true);  // Block align
    view.setUint16(34, 16, true); // 16-bit
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);

    // Write 16-bit PCM samples
    let index = 44;
    for (let i = 0; i < samples.length; i++, index += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: "audio/wav" });
}

function updateRecordingUI(recording) {
    if (recording) {
        micVisualizer.classList.add("recording");
        micIcon.textContent = "⏹️";
        recordLabel.textContent = "Đang Thu... (Thả/Bấm để Dừng)";
        recordStatus.textContent = "Đang lắng nghe giọng Nghệ của bạn...";
    } else {
        micVisualizer.classList.remove("recording");
        micIcon.textContent = "🎙️";
        recordLabel.textContent = "Giữ để Thu Âm";
    }
}

// Support both Hold-to-Record and Click-to-Toggle
recordBtn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    if (!isRecording) {
        holdMode = true;
        startRecording();
    } else {
        holdMode = false;
        stopRecording();
    }
});

window.addEventListener("mouseup", () => {
    if (isRecording && holdMode) {
        const elapsed = performance.now() - recordStartTime;
        // If held for > 450ms, treat as Hold-to-Record release; if quick click, keep recording until next click!
        if (elapsed > 450) {
            holdMode = false;
            stopRecording();
        } else {
            holdMode = false;
        }
    }
});

recordBtn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    if (!isRecording) {
        holdMode = true;
        startRecording();
    } else {
        holdMode = false;
        stopRecording();
    }
});

window.addEventListener("touchend", () => {
    if (isRecording && holdMode) {
        const elapsed = performance.now() - recordStartTime;
        if (elapsed > 450) {
            holdMode = false;
            stopRecording();
        } else {
            holdMode = false;
        }
    }
});

// File upload handler
uploadWavInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    userAudioPlayer.src = url;
    userPlaybackWrap.classList.remove("hidden");
    await uploadAndScoreWav(file, file.name);
    uploadWavInput.value = "";
});

// 4. Send Audio to FastAPI /api/score
async function uploadAndScoreWav(blob, filename) {
    setLoadingState("AI đang phân tích âm học & đối chiếu ngữ điệu Nghệ An...");
    try {
        const formData = new FormData();
        formData.append("file", blob, filename);

        // If practicing an active challenge, attach reference audio for sentence-matched scoring
        if (challenges && challenges[currentChallengeIdx] && challenges[currentChallengeIdx].sample_audio) {
            const sampleAudio = challenges[currentChallengeIdx].sample_audio;
            const refName = sampleAudio.split("/").pop();
            if (refName) {
                formData.append("reference_audio", refName);
            }
        }

        const res = await fetch(`${API_BASE}/api/score`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server trả về lỗi ${res.status}`);
        }

        const result = await res.json();
        displayResult(result);
        recordStatus.textContent = "Đã phân tích xong! Xem đồ thị F0 bên phải để đối chiếu.";
    } catch (err) {
        recordStatus.textContent = "Lỗi chấm điểm: " + err.message;
        rankBadgeEl.textContent = "Lỗi xử lý";
        feedbackTextEl.textContent = err.message;
    }
}

function setLoadingState(msg) {
    recordStatus.textContent = msg;
    rankBadgeEl.textContent = "⏳ Đang phân tích...";
    feedbackTextEl.textContent = "Mô hình AI đang đối chiếu đặc trưng ngữ âm & cao độ chuẩn Nghệ An...";
}

// 5. Render Score & Update Chart
function displayResult(result) {
    const targetScore = Number(result.score) || 0;
    const mseLoss = (result.mse_loss !== undefined && result.mse_loss !== null)
        ? `${result.mse_loss} (NMSE: ${result.nmse ?? "--"})`
        : "Không có giọng nói";

    animateScore(targetScore);
    mseValueEl.textContent = mseLoss;
    rankBadgeEl.textContent = result.rank || "Hoàn thành";
    feedbackTextEl.textContent = result.feedback || "";

    // Color styling based on score tier (Nghe An Countryside Palette)
    let themeColor = "#344E41";
    if (targetScore >= 85) {
        themeColor = "#344E41"; // Xanh lá đậm S-Rank
    } else if (targetScore >= 70) {
        themeColor = "#587A4B"; // Xanh lúa non đậm A-Rank
    } else if (targetScore >= 50) {
        themeColor = "#B87D2B"; // Vàng đồng / Nâu đất B-Rank
    } else {
        themeColor = "#A94F3D"; // Đỏ gạch truyền thống C-Rank
    }

    rankBadgeEl.style.backgroundColor = `${themeColor}18`;
    rankBadgeEl.style.color = themeColor;
    rankBadgeEl.style.border = `1px solid ${themeColor}55`;
    scoreValueEl.style.color = themeColor;

    // Update SVG ring (circumference = 2 * pi * 58 ≈ 365)
    const circumference = 365;
    const offset = circumference - (targetScore / 100) * circumference;
    scoreRingFill.style.strokeDashoffset = offset;
    scoreRingFill.style.stroke = themeColor;

    // Update Chart.js
    if (f0Chart && Array.isArray(result.original_f0) && Array.isArray(result.reconstructed_f0)) {
        f0Chart.data.datasets[0].data = result.original_f0;
        f0Chart.data.datasets[1].data = result.reconstructed_f0;
        f0Chart.update();
    }
}

function animateScore(target) {
    const duration = 900;
    const startTime = performance.now();
    const startVal = 0;

    function tick(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // Cubic ease-out
        const current = (startVal + (target - startVal) * eased).toFixed(1);
        scoreValueEl.textContent = current;
        if (progress < 1) {
            requestAnimationFrame(tick);
        } else {
            scoreValueEl.textContent = target.toFixed(1);
        }
    }

    requestAnimationFrame(tick);
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    initChart();
    loadChallenges();
});
