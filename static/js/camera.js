const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const recalibrateBtn = document.getElementById("recalibrateBtn");
const resetHistoryBtn = document.getElementById("resetHistoryBtn");

const systemStatus = document.getElementById("systemStatus");
const detectionCount = document.getElementById("detectionCount");
const aiStatus = document.getElementById("aiStatus");

const alertBox = document.getElementById("alertBox");
const resultText = document.getElementById("result");
const logList = document.getElementById("logList");

const studentNameInput = document.getElementById("studentName");
const studentIdInput = document.getElementById("studentId");

const calibrationBox = document.getElementById("calibrationBox");
const calibrationBar = document.getElementById("calibrationBar");
const activeFlagsBox = document.getElementById("activeFlags");
const countsBox = document.getElementById("countsBox");
const compactLogBox = document.getElementById("compactLogBox");
const reportLink = document.getElementById("reportLink");

const violationCard = document.getElementById("violationCard");
const historySection = document.getElementById("historySection");
const logFilter = document.getElementById("logFilter");
const viewCompactBtn = document.getElementById("viewCompactBtn");
const viewDetailBtn = document.getElementById("viewDetailBtn");
const loadHistoryBtn = document.getElementById("loadHistoryBtn");
const logSummary = document.getElementById("logSummary");

const metricEls = {
    identity: document.getElementById("mIdentity"),
    refAvatar: document.getElementById("refAvatar"),
    faces: document.getElementById("mFaces"),
    yaw: document.getElementById("mYaw"),
    pitch: document.getElementById("mPitch"),
    gazeH: document.getElementById("mGazeH"),
    gazeV: document.getElementById("mGazeV"),
    blink: document.getElementById("mBlink"),
    image: document.getElementById("mImage")
};

const ctx = canvas.getContext("2d");

// Phân tích đầu/mắt/tư thế ~4 khung hình/giây, YOLO điện thoại ~1 lần/giây
const ANALYZE_INTERVAL_MS = 250;
const PHONE_INTERVAL_MS = 1000;

// Nếu mở trực tiếp file index.html từ PyCharm Built-in Server (cổng 63342) hoặc file://,
// tự động trỏ API và ảnh bằng chứng về máy chủ FastAPI (http://127.0.0.1:8000)
const API_BASE = (window.location.port === "8000") ? "" : "http://127.0.0.1:8000";

function apiUrl(path) {
    if (!path) return path;
    if (path.startsWith("http://") || path.startsWith("https://")) return path;
    return `${API_BASE}${path}`;
}

let cameraStream = null;
let monitoring = false;
let busy = false;
let lastPhoneCheck = 0;
let alertTimer = null;

const capture = document.createElement("canvas");
const captureContext = capture.getContext("2d");

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
    }[c]));
}

async function startCamera() {
    const name = studentNameInput.value.trim();
    const id = studentIdInput.value.trim();

    if (!name || !id) {
        alert("Vui lòng nhập đầy đủ Họ tên và Mã sinh viên trước khi bật camera!");
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        capture.width = video.videoWidth;
        capture.height = video.videoHeight;

        // Bắt đầu phiên mới trên máy chủ (reset + hiệu chuẩn lại)
        const form = new FormData();
        form.append("student_name", name);
        form.append("student_id", id);
        await fetch(apiUrl("/api/session/start"), { method: "POST", body: form });

        monitoring = true;
        lastPhoneCheck = 0;
        allEvents = [];
        renderLogs();

        systemStatus.textContent = "Đang giám sát";
        aiStatus.textContent = "Đang hiệu chuẩn";
        resultText.textContent = "Hãy ngồi thẳng, nhìn vào màn hình để hệ thống hiệu chuẩn...";
        calibrationBox.classList.remove("hidden");
        calibrationBar.style.width = "0%";

        reportLink.href = apiUrl(`/api/report/${encodeURIComponent(id)}/csv`);
        reportLink.classList.remove("hidden");

        startBtn.disabled = true;
        stopBtn.disabled = false;
        recalibrateBtn.disabled = false;
        studentNameInput.disabled = true;
        studentIdInput.disabled = true;

        analyzeCamera();

    } catch (error) {
        alert("Không thể mở camera. Vui lòng kiểm tra quyền truy cập!");
        console.error(error);
    }
}

async function analyzeCamera() {
    if (!monitoring) return;

    if (busy || video.readyState < 2) {
        setTimeout(analyzeCamera, ANALYZE_INTERVAL_MS);
        return;
    }

    busy = true;
    const startedAt = performance.now();

    try {
        captureContext.drawImage(video, 0, 0, capture.width, capture.height);

        const blob = await new Promise(resolve => {
            capture.toBlob(resolve, "image/jpeg", 0.8);
        });

        if (!blob) {
            throw new Error("Không thể chụp hình ảnh");
        }

        const now = Date.now();
        const checkPhone = now - lastPhoneCheck >= PHONE_INTERVAL_MS;
        if (checkPhone) lastPhoneCheck = now;

        const formData = new FormData();
        formData.append("file", blob, "camera.jpg");
        formData.append("student_name", studentNameInput.value);
        formData.append("student_id", studentIdInput.value);
        formData.append("check_phone", checkPhone ? "true" : "false");

        const response = await fetch(apiUrl("/api/detect"), {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Lỗi từ máy chủ AI");
        }

        const data = await response.json();
        if (!monitoring) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawOverlay(data.overlay, data.active_flags);
        if (data.phone_detected) {
            drawDetections(data.detections);
        }

        updateCalibration(data.calibration);
        updateMetrics(data.metrics, data.calibration);
        updateFlags(data.active_flags);
        updateCounts(data.counts);
        detectionCount.textContent = data.total_violations;

        if (data.violations && data.violations.length > 0) {
            showAlert(data.violations, data.evidence_image, data.detections);
        } else if (data.calibration.done) {
            resultText.textContent = data.active_flags.length
                ? "Đang theo dõi hành vi nghi ngờ..."
                : "Chưa phát hiện hành vi gian lận.";
        }

    } catch (error) {
        console.error(error);
        resultText.textContent = "Không thể kết nối máy chủ AI.";
    } finally {
        busy = false;
        if (monitoring) {
            const elapsed = performance.now() - startedAt;
            setTimeout(analyzeCamera, Math.max(0, ANALYZE_INTERVAL_MS - elapsed));
        }
    }
}

// Canvas bị lật ngang bằng CSS (giống gương) nên chữ phải lật lại để đọc được
function drawText(text, x, y, color) {
    ctx.save();
    ctx.scale(-1, 1);
    ctx.font = "bold 18px Arial";
    const width = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(-x - width - 4, y - 18, width + 8, 24);
    ctx.fillStyle = color;
    ctx.fillText(text, -x - width, y);
    ctx.restore();
}

function drawOverlay(overlay, activeFlags) {
    if (!overlay) return;
    const suspicious = activeFlags && activeFlags.length > 0;

    overlay.faces.forEach((face, index) => {
        const [x1, y1, x2, y2] = face.bbox;
        let boxColor = "#22c55e";
        if (face.is_registered === false) {
            boxColor = "#ef4444"; // Đỏ: Người lạ / sai danh tính
        } else if (suspicious) {
            boxColor = "#facc15";
        } else if (index > 0) {
            boxColor = "#f97316";
        }
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        if (face.label) {
            drawText(
                face.label,
                x2,
                Math.max(22, y1 - 6),
                face.is_registered === false ? "#f87171" : "#4ade80"
            );
        }

        // Hướng đầu (mũi tên từ chóp mũi)
        ctx.strokeStyle = "#e879f9";
        ctx.beginPath();
        ctx.moveTo(face.nose[0], face.nose[1]);
        ctx.lineTo(face.dir[0], face.dir[1]);
        ctx.stroke();

        // Mống mắt
        ctx.fillStyle = "#22d3ee";
        face.irises.forEach(p => {
            ctx.beginPath();
            ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
            ctx.fill();
        });
    });

    overlay.poses.forEach(pose => {
        const [a, b] = pose.shoulders;
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        ctx.fillStyle = "#3b82f6";
        pose.wrists.forEach(w => {
            ctx.beginPath();
            ctx.arc(w[0], w[1], 6, 0, Math.PI * 2);
            ctx.fill();
        });
    });

    if (suspicious) {
        activeFlags.slice(0, 3).forEach((flag, i) => {
            const suffix = flag.latched
                ? `✓ Đã ghi nhận (${flag.duration}s)`
                : `${flag.duration}/${flag.required}s`;
            drawText(`${flag.label} · ${suffix}`,
                10, 28 + i * 28, flag.severity === "high" ? "#f87171" : "#facc15");
        });
    }
}

function drawDetections(detections) {
    detections.forEach(item => {
        const x = item.x1;
        const y = item.y1;
        const width = item.x2 - item.x1;
        const height = item.y2 - item.y1;

        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        drawText(
            "ĐIỆN THOẠI " + Math.round(item.confidence * 100) + "%",
            x,
            Math.max(20, y - 8),
            "#ff4d4f"
        );
    });
}

function updateCalibration(calibration) {
    if (calibration.done) {
        if (calibration.avatar_url && metricEls.refAvatar) {
            const fullUrl = apiUrl(calibration.avatar_url);
            if (metricEls.refAvatar.dataset.src !== fullUrl) {
                metricEls.refAvatar.src = fullUrl;
                metricEls.refAvatar.dataset.src = fullUrl;
                metricEls.refAvatar.classList.remove("hidden");
            }
        }
        if (!calibrationBox.classList.contains("hidden")) {
            calibrationBox.classList.add("hidden");
            aiStatus.textContent = "Đang phân tích";
            resultText.textContent = "Hiệu chuẩn & đăng ký khuôn mặt xong. Bắt đầu giám sát.";
        }
    } else {
        calibrationBox.classList.remove("hidden");
        calibrationBar.style.width = Math.round(calibration.progress * 100) + "%";
        if (metricEls.identity) {
            metricEls.identity.textContent = `Đang lấy mẫu khuôn mặt (${Math.round(calibration.progress * 100)}%)...`;
            metricEls.identity.className = "identity-status";
        }
    }
}

function fmtDir(value, unit, negLabel, posLabel, digits = 0) {
    if (value === undefined || value === null) return "-";
    const label = Math.abs(value) < (unit === "°" ? 5 : 0.05) ? "Thẳng" : (value > 0 ? posLabel : negLabel);
    return `${label} ${Math.abs(value).toFixed(digits)}${unit}`;
}

function updateMetrics(m, calibration) {
    metricEls.faces.textContent = `${m.faces}`;
    metricEls.image.textContent = `${m.brightness} / ${m.sharpness}`;

    if (calibration.done && metricEls.identity) {
        if (m.faces === 0) {
            metricEls.identity.textContent = "Không thấy khuôn mặt";
            metricEls.identity.className = "identity-status danger";
        } else if (m.identity_matched) {
            metricEls.identity.textContent = `✓ Đúng thí sinh (${m.identity_similarity}% khớp)`;
            metricEls.identity.className = "identity-status ok";
        } else if (m.identity_matched === false) {
            metricEls.identity.textContent = `⚠️ NGƯỜI LẠ (${m.identity_similarity}% khớp)`;
            metricEls.identity.className = "identity-status danger";
        }
    }

    if (m.yaw === undefined) {
        ["yaw", "pitch", "gazeH", "gazeV", "blink"].forEach(k => metricEls[k].textContent = "-");
    } else if (!calibration.done) {
        metricEls.yaw.textContent = `${m.yaw}° (thô)`;
        metricEls.pitch.textContent = `${m.pitch}° (thô)`;
        metricEls.gazeH.textContent = m.gaze_h;
        metricEls.gazeV.textContent = m.gaze_v;
        metricEls.blink.textContent = `${m.seconds_since_blink}s`;
    } else {
        // Quy ước khớp với màn hình mirror: d_yaw > 0 là Phải màn hình, d_gaze_h > 0 là Phải màn hình
        metricEls.yaw.textContent = fmtDir(m.d_yaw, "°", "Trái", "Phải");
        metricEls.pitch.textContent = fmtDir(m.d_pitch, "°", "Ngửa", "Cúi");
        metricEls.gazeH.textContent = fmtDir(m.d_gaze_h, "", "Trái", "Phải", 2);
        metricEls.gazeV.textContent = fmtDir(m.d_gaze_v, "", "Lên", "Xuống", 2);
        metricEls.blink.textContent = m.eyes_closed ? "Đang nhắm" : `${m.seconds_since_blink}s trước`;
    }
}

function updateFlags(flags) {
    if (!flags || flags.length === 0) {
        activeFlagsBox.textContent = "Không có";
        return;
    }
    activeFlagsBox.innerHTML = flags.map(f => {
        const tag = f.latched ? `✓ Đã báo (${f.duration}s)` : `${f.duration}/${f.required}s`;
        return `<span class="flag ${f.severity === "high" ? "high" : ""}">${escapeHtml(f.label)} · ${tag}</span>`;
    }).join("");
}

const RULE_LABELS = {};
let allEvents = [];
let logViewMode = "compact"; // "compact" | "detail"
let currentFilter = "ALL";

const SEVERITY_TEXT = {
    high: "Nghiêm trọng",
    medium: "Cảnh báo",
    low: "Nhắc nhở"
};

function formatShortTime(isoOrTime) {
    if (!isoOrTime) return "--:--:--";
    if (isoOrTime.includes("T")) {
        return isoOrTime.split("T")[1].slice(0, 8);
    }
    return isoOrTime;
}

function updateCounts(counts) {
    const entries = Object.entries(counts || {});
    if (entries.length === 0) {
        countsBox.textContent = "Chưa có";
        return;
    }
    countsBox.innerHTML = entries.map(([code, n]) =>
        `<span class="flag count">${escapeHtml(RULE_LABELS[code] || code)}: ${n}</span>`
    ).join("");
}

function syncFilterOptions() {
    const prev = logFilter.value || "ALL";
    const codes = Array.from(new Set(allEvents.map(e => e.code)));
    logFilter.innerHTML = `<option value="ALL">Tất cả loại vi phạm (${allEvents.length})</option>` +
        codes.map(c => {
            const count = allEvents.filter(e => e.code === c).length;
            const label = RULE_LABELS[c] || c;
            return `<option value="${escapeHtml(c)}">${escapeHtml(label)} (${count})</option>`;
        }).join("");
    logFilter.value = codes.includes(prev) ? prev : "ALL";
    currentFilter = logFilter.value;
}

function renderSummaryBar() {
    if (allEvents.length === 0) {
        logSummary.classList.add("hidden");
        return;
    }
    logSummary.classList.remove("hidden");
    const countsByCode = {};
    allEvents.forEach(e => {
        countsByCode[e.code] = (countsByCode[e.code] || 0) + 1;
    });
    const chips = [
        `<span class="summary-chip ${currentFilter === "ALL" ? "active" : ""}" data-code="ALL">Tổng: ${allEvents.length}</span>`
    ];
    Object.entries(countsByCode).forEach(([code, count]) => {
        const active = currentFilter === code ? "active" : "";
        chips.push(
            `<span class="summary-chip ${active}" data-code="${escapeHtml(code)}">${escapeHtml(RULE_LABELS[code] || code)}: ${count}</span>`
        );
    });
    logSummary.innerHTML = chips.join("");
    logSummary.querySelectorAll(".summary-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            currentFilter = chip.dataset.code;
            logFilter.value = currentFilter;
            renderLogs();
        });
    });
}

function renderCompactSidebar() {
    if (allEvents.length === 0) {
        compactLogBox.textContent = "Chưa có vi phạm";
        return;
    }
    // Hiển thị tối đa 30 dòng mới nhất ở bảng bên cạnh camera
    const latest = allEvents.slice().reverse().slice(0, 30);
    compactLogBox.innerHTML = latest.map((ev, idx) => {
        const num = allEvents.length - idx;
        const t = escapeHtml(formatShortTime(ev.time));
        const sev = ev.severity || "medium";
        const link = ev.evidence_image
            ? `<a class="compact-log-link" href="${apiUrl(ev.evidence_image)}" target="_blank">[Ảnh]</a>`
            : "";
        return `
            <div class="compact-log-row">
                <span class="compact-log-time">#${num} ${t}</span>
                <span class="compact-log-msg ${sev}" title="${escapeHtml(ev.label)} (${ev.duration}s)">${escapeHtml(ev.label)} (${ev.duration}s)</span>
                ${link}
            </div>
        `;
    }).join("");
}

function renderLogs() {
    syncFilterOptions();
    renderSummaryBar();
    renderCompactSidebar();

    const filtered = currentFilter === "ALL"
        ? allEvents.slice().reverse()
        : allEvents.filter(e => e.code === currentFilter).reverse();

    if (filtered.length === 0) {
        logList.textContent = "Chưa có dữ liệu vi phạm.";
        return;
    }

    if (logViewMode === "compact") {
        const rows = filtered.map(ev => {
            const t = escapeHtml(formatShortTime(ev.time));
            const sev = ev.severity || "medium";
            const sevLabel = SEVERITY_TEXT[sev] || sev;
            const evidenceCell = ev.evidence_image
                ? `<a class="thumb-preview" href="${apiUrl(ev.evidence_image)}" target="_blank">
                       <img src="${apiUrl(ev.evidence_image)}" alt="Bằng chứng" loading="lazy" /> Xem ảnh
                   </a>`
                : "-";
            return `
                <tr>
                    <td><b>#${ev.seq}</b></td>
                    <td>${t}</td>
                    <td><b>${escapeHtml(ev.label)}</b></td>
                    <td>${ev.duration}s</td>
                    <td><span class="badge-sev ${sev}">${sevLabel}</span></td>
                    <td>${escapeHtml(ev.student_name)} (${escapeHtml(ev.student_id)})</td>
                    <td>${evidenceCell}</td>
                </tr>
            `;
        }).join("");

        logList.innerHTML = `
            <table class="log-table">
                <thead>
                    <tr>
                        <th>Lần</th>
                        <th>Thời gian</th>
                        <th>Hành vi vi phạm</th>
                        <th>Kéo dài</th>
                        <th>Mức độ</th>
                        <th>Thí sinh</th>
                        <th>Bằng chứng</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    } else {
        logList.innerHTML = filtered.map(ev => {
            const t = escapeHtml(formatShortTime(ev.time));
            const sev = ev.severity || "medium";
            const imgHtml = ev.evidence_image
                ? `- Bằng chứng hình ảnh: <br><a href="${apiUrl(ev.evidence_image)}" target="_blank"><img src="${apiUrl(ev.evidence_image)}" alt="Bằng chứng vi phạm" style="max-width: 220px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc;" /></a>`
                : "";
            return `
                <div class="log-item severity-${sev}" style="border: 1px solid #ff4d4f; padding: 10px; margin-bottom: 10px; border-radius: 5px; background: #fff1f0;">
                    <strong>⚠️ Lần #${ev.seq} — Phát hiện lúc ${t}</strong><br>
                    - Thí sinh: <b>${escapeHtml(ev.student_name)}</b> (${escapeHtml(ev.student_id)})<br>
                    - Hành vi: <b>${escapeHtml(ev.label)}</b><br>
                    - Thời gian kéo dài: <b>${ev.duration}s</b><br>
                    ${imgHtml}
                </div>
            `;
        }).join("");
    }
}

function showAlert(violations, evidenceUrl, detections) {
    violations.forEach(v => RULE_LABELS[v.code] = v.label);
    const labels = violations.map(v => v.label).join(", ");

    alertBox.textContent = `⚠️ CẢNH BÁO: PHÁT HIỆN HÀNH VI GIAN LẬN (${labels.toUpperCase()})!`;
    alertBox.classList.remove("hidden");
    clearTimeout(alertTimer);
    alertTimer = setTimeout(() => alertBox.classList.add("hidden"), 4000);

    resultText.textContent = `${labels} - Đã lưu bằng chứng!`;

    const time = new Date().toLocaleTimeString("vi-VN");
    const studentName = studentNameInput.value.trim();
    const studentId = studentIdInput.value.trim();

    violations.forEach(v => {
        allEvents.push({
            seq: allEvents.length + 1,
            time: v.time || time,
            code: v.code,
            label: v.label,
            severity: v.severity || "medium",
            duration: v.duration,
            evidence_image: v.evidence_image || evidenceUrl,
            student_name: studentName,
            student_id: studentId
        });
    });

    renderLogs();
}

async function loadSavedHistory() {
    const id = studentIdInput.value.trim();
    if (!id) {
        alert("Vui lòng nhập Mã sinh viên (MSSV) để tải lịch sử vi phạm!");
        return;
    }
    try {
        const res = await fetch(apiUrl(`/api/report/${encodeURIComponent(id)}`));
        if (!res.ok) throw new Error("Không tải được lịch sử");
        const data = await res.json();
        const events = data.events || [];
        allEvents = events.map((ev, i) => {
            if (ev.code && ev.label) RULE_LABELS[ev.code] = ev.label;
            return {
                seq: i + 1,
                time: ev.time,
                code: ev.code,
                label: ev.label || ev.code,
                severity: ev.severity || "medium",
                duration: ev.duration ?? 0,
                evidence_image: ev.evidence_image,
                student_name: ev.student_name || studentNameInput.value.trim() || "Chưa rõ",
                student_id: ev.student_id || id
            };
        });
        detectionCount.textContent = allEvents.length;

        const countsByCode = {};
        allEvents.forEach(e => { countsByCode[e.code] = (countsByCode[e.code] || 0) + 1; });
        updateCounts(countsByCode);

        reportLink.href = apiUrl(`/api/report/${encodeURIComponent(id)}/csv`);
        if (allEvents.length > 0) {
            reportLink.classList.remove("hidden");
        }
        renderLogs();
    } catch (err) {
        console.error(err);
    }
}

async function stopCamera() {
    monitoring = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => {
            track.stop();
        });
        cameraStream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    alertBox.classList.add("hidden");
    calibrationBox.classList.add("hidden");
    systemStatus.textContent = "Đã tắt camera";
    aiStatus.textContent = "Sẵn sàng";
    resultText.textContent = "Buổi giám sát đã dừng.";

    startBtn.disabled = false;
    stopBtn.disabled = true;
    recalibrateBtn.disabled = true;
    studentNameInput.disabled = false;
    studentIdInput.disabled = false;

    try {
        const form = new FormData();
        form.append("student_id", studentIdInput.value);
        const response = await fetch(apiUrl("/api/session/stop"), { method: "POST", body: form });
        const data = await response.json();
        if (data.summary) {
            const s = data.summary;
            resultText.textContent =
                `Buổi giám sát đã dừng. Thời lượng ${s.duration_seconds}s, ` +
                `${s.frames_analyzed} khung hình, ${s.total_violations} vi phạm, ${s.blinks} lần chớp mắt.`;
        }
    } catch (error) {
        console.error(error);
    }
}

async function recalibrateCamera() {
    if (!monitoring) return;
    const form = new FormData();
    form.append("student_name", studentNameInput.value.trim());
    form.append("student_id", studentIdInput.value.trim());
    await fetch(apiUrl("/api/session/recalibrate"), { method: "POST", body: form });
    aiStatus.textContent = "Đang hiệu chuẩn";
    resultText.textContent = "Hãy ngồi thẳng, nhìn vào màn hình để hiệu chuẩn lại...";
    calibrationBox.classList.remove("hidden");
    calibrationBar.style.width = "0%";
}

async function resetHistory() {
    const id = studentIdInput.value.trim();
    const name = studentNameInput.value.trim() || "Chưa rõ";
    if (!id) {
        alert("Vui lòng nhập Mã sinh viên (MSSV) cần xóa lịch sử!");
        return;
    }
    if (!confirm(`Bạn có chắc muốn xóa toàn bộ lịch sử vi phạm và ảnh bằng chứng của MSSV "${id}"?`)) {
        return;
    }
    try {
        const form = new FormData();
        form.append("student_name", name);
        form.append("student_id", id);
        await fetch(apiUrl("/api/session/reset"), { method: "POST", body: form });
        allEvents = [];
        detectionCount.textContent = "0";
        updateCounts({});
        renderLogs();
        resultText.textContent = `Đã xóa sạch lịch sử vi phạm của MSSV ${id}.`;
    } catch (err) {
        console.error(err);
    }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
recalibrateBtn.addEventListener("click", recalibrateCamera);
resetHistoryBtn.addEventListener("click", resetHistory);

violationCard.addEventListener("click", () => {
    historySection.scrollIntoView({ behavior: "smooth" });
});

logFilter.addEventListener("change", () => {
    currentFilter = logFilter.value;
    renderLogs();
});

viewCompactBtn.addEventListener("click", () => {
    logViewMode = "compact";
    viewCompactBtn.classList.add("active");
    viewDetailBtn.classList.remove("active");
    renderLogs();
});

viewDetailBtn.addEventListener("click", () => {
    logViewMode = "detail";
    viewDetailBtn.classList.add("active");
    viewCompactBtn.classList.remove("active");
    renderLogs();
});

loadHistoryBtn.addEventListener("click", loadSavedHistory);
