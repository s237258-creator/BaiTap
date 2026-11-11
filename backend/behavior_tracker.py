"""
Theo dõi hành vi theo THỜI GIAN cho từng thí sinh (phiên giám sát).

Điểm cải tiến chính:
1. Cơ chế One-shot Latch (mỗi đợt vi phạm chỉ báo 1 lần duy nhất):
   - Khi một hành vi kéo dài vượt ngưỡng `min_dur`, hệ thống ghi nhận 1 lần vi phạm
     và khoá (latch) hành vi đó lại.
   - Chỉ khi thí sinh trở về tư thế bình thường liên tục ít nhất `CLEAR_RELATCH_SECONDS`
     (0.8 giây) thì khoá mới mở để tính lần vi phạm tiếp theo nếu tái phạm.
2. Quy ước Trái / Phải khớp trực tiếp với hướng trên màn hình camera (mirror).
3. Hỗ trợ hiệu chuẩn lại (recalibrate) và xoá sạch lịch sử vi phạm của thí sinh.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from backend import config
from backend.behavior_analyzer import FrameFeatures

try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT_PATHS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"]
except ImportError:  # pragma: no cover
    Image = None


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip())[:50]
    return cleaned or "unknown"


# ---------------------------------------------------------------------------
# Bộ đếm thời gian cho từng luật (One-shot Latch)
# ---------------------------------------------------------------------------
@dataclass
class RuleTimer:
    start: float | None = None
    last_seen: float = 0.0
    last_fired: float = -1e9
    latched: bool = False          # Đã báo trong đợt hành vi hiện tại chưa
    clear_since: float | None = None

    def update(self, condition: bool | None, now: float, min_dur: float, cooldown: float) -> tuple[bool, float]:
        """
        Trả về (có vi phạm mới không, thời gian hành vi đang kéo dài).
        Nếu đã báo (latched=True), giữ nguyên trạng thái đã báo và KHÔNG bắn thêm
        cho đến khi condition=False liên tục >= CLEAR_RELATCH_SECONDS.
        """
        if condition is None:
            if self.start is not None and now - self.last_seen > config.MAX_FRAME_GAP:
                self.start = None
                self.latched = False
                self.clear_since = None
            return False, (self.last_seen - self.start) if self.start is not None else 0.0

        if not condition:
            if self.latched:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= config.CLEAR_RELATCH_SECONDS:
                    self.latched = False
                    self.start = None
                    self.clear_since = None
            else:
                self.start = None
            return False, 0.0

        # condition == True
        self.clear_since = None
        if self.start is None or now - self.last_seen > config.MAX_FRAME_GAP:
            self.start = now
            self.latched = False
        self.last_seen = now
        duration = now - self.start

        if not self.latched and duration >= min_dur and now - self.last_fired >= cooldown:
            self.last_fired = now
            self.latched = True
            return True, duration

        return False, duration


@dataclass
class Baseline:
    yaw: float = 0.0
    pitch: float = 0.0
    gaze_h: float = 0.5
    gaze_v: float = 0.5
    gaze_score_h: float = 0.0
    ear: float = 0.27
    look_down: float = 0.35
    embedding: np.ndarray | None = None
    avatar_url: str | None = None


@dataclass
class SessionState:
    student_id: str
    student_name: str
    started_at: float = field(default_factory=time.time)
    calibrated: bool = False
    calib_start: float | None = None
    calib_samples: list[dict] = field(default_factory=list)
    baseline: Baseline = field(default_factory=Baseline)
    timers: dict[str, RuleTimer] = field(default_factory=dict)
    smooth: dict[str, float] = field(default_factory=dict)
    prev_gray: np.ndarray | None = None
    prev_landmarks: np.ndarray | None = None
    last_blink: float = field(default_factory=time.time)
    eyes_closed: bool = False
    blink_count: int = 0
    frames: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def timer(self, code: str) -> RuleTimer:
        return self.timers.setdefault(code, RuleTimer())


_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def start_session(student_id: str, student_name: str) -> SessionState:
    with _sessions_lock:
        state = SessionState(student_id=student_id, student_name=student_name)
        _sessions[student_id] = state
        return state


def get_session(student_id: str, student_name: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(student_id)
        if state is None:
            state = SessionState(student_id=student_id, student_name=student_name)
            _sessions[student_id] = state
        return state


def recalibrate_session(student_id: str, student_name: str) -> SessionState:
    """Đặt lại quá trình hiệu chuẩn 3 giây mà không làm mất lịch sử vi phạm hiện có."""
    state = get_session(student_id, student_name)
    with state.lock:
        state.calibrated = False
        state.calib_start = None
        state.calib_samples.clear()
        state.timers.clear()
        state.smooth.clear()
        state.last_blink = time.time()
    return state


def reset_student_history(student_id: str, student_name: str = "Chưa rõ") -> None:
    """Xoá sạch số đếm, nhật ký CSV/JSONL và ảnh bằng chứng của thí sinh."""
    sid = safe_id(student_id)
    with _sessions_lock:
        state = _sessions.get(student_id)
        if state is not None:
            with state.lock:
                state.counts.clear()
                state.events.clear()
                state.timers.clear()
        else:
            _sessions[student_id] = SessionState(student_id=student_id, student_name=student_name)

    ev_dir = config.EVIDENCE_DIR / sid
    rep_dir = config.REPORTS_DIR / sid
    if ev_dir.exists():
        shutil.rmtree(ev_dir, ignore_errors=True)
    if rep_dir.exists():
        shutil.rmtree(rep_dir, ignore_errors=True)


def end_session(student_id: str) -> dict | None:
    with _sessions_lock:
        state = _sessions.pop(student_id, None)
    return summarize(state) if state else None


def summarize(state: SessionState) -> dict:
    return {
        "student_id": state.student_id,
        "student_name": state.student_name,
        "started_at": datetime.fromtimestamp(state.started_at).isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - state.started_at, 1),
        "frames_analyzed": state.frames,
        "blinks": state.blink_count,
        "total_violations": sum(state.counts.values()),
        "violations_by_type": {
            code: {"label": config.RULES[code][0], "count": n}
            for code, n in state.counts.items() if code in config.RULES
        },
        "events": state.events,
    }


# ---------------------------------------------------------------------------
# Xử lý một khung hình
# ---------------------------------------------------------------------------
def _ema(state: SessionState, key: str, value: float) -> float:
    prev = state.smooth.get(key)
    a = config.SMOOTHING_ALPHA
    new = value if prev is None else a * value + (1 - a) * prev
    state.smooth[key] = new
    return new


def _median(samples: list[dict], key: str) -> float:
    return float(np.median([s[key] for s in samples]))


def _save_reference_avatar(state: SessionState, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    sid = safe_id(state.student_id)
    folder = config.EVIDENCE_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * 0.22)
    pad_y = int((y2 - y1) * 0.22)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = cv2.flip(frame[cy1:cy2, cx1:cx2], 1)
    filename = f"reference_avatar_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(folder / filename), crop)
    return f"/static/evidence/{sid}/{filename}"


def _update_calibration(state: SessionState, frame: np.ndarray | None, feats: FrameFeatures, now: float) -> float:
    """Thu thập mẫu khi thí sinh nhìn thẳng; trả về tiến độ 0..1."""
    face = feats.main_face
    valid = (
        len(feats.faces) == 1
        and abs(face.yaw) <= config.CALIBRATION_MAX_ABS_YAW
        and abs(face.pitch) <= config.CALIBRATION_MAX_ABS_PITCH
        and face.blink_score < config.BLINK_BLENDSHAPE
    )
    if not valid:
        if state.calib_start is None:
            return 0.0
        return min((now - state.calib_start) / config.CALIBRATION_SECONDS,
                   len(state.calib_samples) / config.CALIBRATION_MIN_SAMPLES, 0.99)

    if state.calib_start is None:
        state.calib_start = now
    sample = {
        "yaw": face.yaw, "pitch": face.pitch,
        "gaze_h": face.gaze_h, "gaze_v": face.gaze_v,
        "gaze_score_h": face.gaze_score_h,
        "ear": face.ear, "look_down": face.look_down_score,
        "embedding": face.embedding,
    }
    state.calib_samples.append(sample)

    elapsed = now - state.calib_start
    progress = min(elapsed / config.CALIBRATION_SECONDS,
                   len(state.calib_samples) / config.CALIBRATION_MIN_SAMPLES, 1.0)
    if progress >= 1.0:
        s = state.calib_samples
        embs = [x["embedding"] for x in s if x.get("embedding") is not None]
        mean_emb = None
        if embs:
            avg = np.mean(np.stack(embs, axis=0), axis=0)
            norm = float(np.linalg.norm(avg))
            if norm > 1e-6:
                mean_emb = (avg / norm).astype(np.float32)

        avatar_url = None
        if frame is not None:
            avatar_url = _save_reference_avatar(state, frame, face.bbox)

        state.baseline = Baseline(
            yaw=_median(s, "yaw"),
            pitch=_median(s, "pitch"),
            gaze_h=_median(s, "gaze_h"),
            gaze_v=_median(s, "gaze_v"),
            gaze_score_h=_median(s, "gaze_score_h"),
            ear=_median(s, "ear"),
            look_down=_median(s, "look_down"),
            embedding=mean_emb,
            avatar_url=avatar_url,
        )
        state.calibrated = True
        state.last_blink = now
    return progress


def process_frame(
    state: SessionState,
    frame: np.ndarray,
    feats: FrameFeatures,
    phone_detections: list[dict] | None,
) -> dict:
    """phone_detections = None nghĩa là khung hình này không chạy YOLO."""
    now = time.time()
    with state.lock:
        state.frames += 1
        face = feats.main_face
        b = state.baseline
        conditions: dict[str, bool | None] = {}
        metrics: dict[str, float | int | None] = {
            "faces": len(feats.faces), "persons": len(feats.poses),
            "brightness": round(feats.brightness, 1), "sharpness": round(feats.sharpness, 1),
        }

        # ---------------- Chất lượng ảnh / gian lận hình ảnh ----------------
        blocked = feats.brightness < config.DARK_BRIGHTNESS or feats.contrast < config.LOW_CONTRAST_STD
        conditions["CAMERA_BLOCKED"] = blocked

        frozen = False
        if state.prev_gray is not None and feats.small_gray is not None and not blocked:
            pixel_diff = float(cv2.absdiff(state.prev_gray, feats.small_gray).mean())
            metrics["pixel_diff"] = round(pixel_diff, 2)
            if face is not None and state.prev_landmarks is not None \
                    and state.prev_landmarks.shape == face.landmarks_px.shape:
                lm_diff = float(np.linalg.norm(face.landmarks_px - state.prev_landmarks, axis=1).mean())
                metrics["landmark_diff"] = round(lm_diff, 2)
                frozen = pixel_diff < config.FROZEN_PIXEL_DIFF and lm_diff < config.FROZEN_LANDMARK_DIFF
            else:
                frozen = pixel_diff < config.FROZEN_PIXEL_DIFF * 0.3
        conditions["FROZEN_FRAME"] = frozen
        state.prev_gray = feats.small_gray
        state.prev_landmarks = face.landmarks_px if face is not None else None

        # ---------------- Điện thoại / số người / rời khung hình ----------------
        conditions["PHONE"] = None if phone_detections is None else len(phone_detections) > 0
        # Chỉ tính nhiều người khi thực sự phát hiện >= 2 khuôn mặt (tránh Pose bắt nhầm rèm/áo)
        conditions["MULTIPLE_PEOPLE"] = len(feats.faces) >= 2
        conditions["NO_FACE"] = face is None and not blocked

        # ---------------- Hiệu chuẩn ----------------
        progress = 1.0
        if not state.calibrated and face is not None:
            progress = _update_calibration(state, frame, feats, now)
        elif not state.calibrated:
            progress = 0.0

        # ---------------- Xác minh danh tính (SFace) ----------------
        target_face = face
        wrong_person = False
        if state.calibrated and b.embedding is not None and feats.faces:
            matched_candidates = []
            for f in feats.faces:
                if f.embedding is not None:
                    sim = float(np.dot(f.embedding, b.embedding))
                    f.similarity = round(max(0.0, sim) * 100.0, 1)
                    f.is_registered = sim >= config.IDENTITY_COSINE_THRESHOLD
                    if f.is_registered:
                        matched_candidates.append((sim, f))
                else:
                    f.similarity = None
                    f.is_registered = None

            if matched_candidates:
                # Khóa đúng vào khuôn mặt của thí sinh đã đăng ký (dù có người lạ to hơn đứng cạnh)
                matched_candidates.sort(key=lambda item: item[0], reverse=True)
                best_sim, target_face = matched_candidates[0]
                metrics["identity_similarity"] = round(max(0.0, best_sim) * 100.0, 1)
                metrics["identity_matched"] = True
            else:
                # Không có khuôn mặt nào trong khung hình khớp với thí sinh đã đăng ký
                target_face = None
                sims = [f.similarity for f in feats.faces if f.similarity is not None]
                metrics["identity_similarity"] = max(sims) if sims else 0.0
                metrics["identity_matched"] = False
                # Chỉ kết luận Sai danh tính / Thi hộ khi khuôn mặt hiện tại nhìn tương đối thẳng
                # (tránh báo nhầm khi thí sinh chính đang quay đầu quá nghiêng hoặc cúi gằm mặt)
                any_frontal_stranger = any(
                    f.similarity is not None
                    and abs(f.yaw - b.yaw) <= config.IDENTITY_MAX_CHECK_YAW
                    and abs(f.pitch - b.pitch) <= config.IDENTITY_MAX_CHECK_PITCH
                    for f in feats.faces
                )
                wrong_person = any_frontal_stranger

        conditions["WRONG_PERSON"] = wrong_person

        # ---------------- Đầu, mắt, chớp mắt (CHỈ tính trên khuôn mặt của thí sinh đã đăng ký) ----------------
        if target_face is None:
            state.smooth.clear()
        else:
            yaw = _ema(state, "yaw", target_face.yaw)
            pitch = _ema(state, "pitch", target_face.pitch)
            gaze_h = _ema(state, "gaze_h", target_face.gaze_h)
            gaze_v = _ema(state, "gaze_v", target_face.gaze_v)
            gaze_score_h = _ema(state, "gaze_score_h", target_face.gaze_score_h)
            look_down = _ema(state, "look_down", target_face.look_down_score)

            metrics.update(yaw=round(yaw, 1), pitch=round(pitch, 1), roll=round(target_face.roll, 1),
                           gaze_h=round(gaze_h, 3), gaze_v=round(gaze_v, 3), ear=round(target_face.ear, 3))

            closed = target_face.blink_score > config.BLINK_BLENDSHAPE or (
                state.calibrated and target_face.ear < config.BLINK_EAR_RATIO * b.ear)
            if closed and not state.eyes_closed:
                state.blink_count += 1
                state.last_blink = now
            state.eyes_closed = closed
            metrics["eyes_closed"] = closed
            metrics["seconds_since_blink"] = round(now - state.last_blink, 1)

            if state.calibrated:
                # Tất cả quy ước hướng theo màn hình hiển thị (mirror):
                #   d_yaw > 0  -> quay sang PHẢI màn hình
                #   d_yaw < 0  -> quay sang TRÁI màn hình
                #   d_pitch > 0 -> cúi đầu xuống
                d_yaw = yaw - b.yaw
                d_pitch = pitch - b.pitch
                # d_gaze_x > 0 -> liếc sang PHẢI màn hình; < 0 -> liếc sang TRÁI màn hình
                d_iris_right = -(gaze_h - b.gaze_h)
                d_score_right = gaze_score_h - b.gaze_score_h
                d_gv = gaze_v - b.gaze_v
                d_look_down = look_down - b.look_down

                metrics.update(
                    d_yaw=round(d_yaw, 1),
                    d_pitch=round(d_pitch, 1),
                    d_gaze_h=round(d_score_right, 2),
                    d_gaze_v=round(d_look_down, 2),
                )

                # 1) Quay đầu Trái / Phải / Cúi đầu
                conditions["HEAD_RIGHT"] = d_yaw > config.HEAD_YAW_THRESHOLD
                conditions["HEAD_LEFT"] = d_yaw < -config.HEAD_YAW_THRESHOLD
                conditions["HEAD_DOWN"] = d_pitch > config.HEAD_PITCH_DOWN_THRESHOLD

                # 2) Liếc mắt (chỉ xét khi đầu tương đối thẳng và mắt mở)
                head_straight = (
                    abs(d_yaw) < config.GAZE_MAX_HEAD_YAW
                    and abs(d_pitch) < config.GAZE_MAX_HEAD_PITCH
                )
                eyes_ok = head_straight and not closed
                conditions["GAZE_RIGHT"] = eyes_ok and (
                    d_score_right > config.GAZE_SCORE_H_THRESHOLD
                    or d_iris_right > config.GAZE_IRIS_H_THRESHOLD
                )
                conditions["GAZE_LEFT"] = eyes_ok and (
                    d_score_right < -config.GAZE_SCORE_H_THRESHOLD
                    or d_iris_right < -config.GAZE_IRIS_H_THRESHOLD
                )
                # Liếc mắt xuống: cả blendshape eyeLookDown lẫn vị trí mống mắt phải hạ xuống rõ rệt
                conditions["GAZE_DOWN"] = eyes_ok and (
                    d_look_down > config.GAZE_DOWN_SCORE_THRESHOLD
                    and (d_gv > config.GAZE_IRIS_V_DOWN_THRESHOLD or look_down > 0.65)
                )
                # Không chớp mắt quá lâu (nghi dùng ảnh tĩnh / video giả)
                conditions["NO_BLINK"] = not closed

        # ---------------- Áp luật thời gian (One-shot Latch) ----------------
        new_events, active = [], []
        for code, (label, min_dur, cooldown, severity) in config.RULES.items():
            cond = conditions.get(code, False)
            timer = state.timer(code)
            if code == "NO_BLINK" and state.calibrated and target_face is not None:
                # Với NO_BLINK, tính trực tiếp theo thời gian kể từ lần chớp mắt gần nhất
                no_blink_dur = now - state.last_blink
                if closed:
                    timer.latched = False
                fired = False
                if not closed and not timer.latched and no_blink_dur >= min_dur and now - timer.last_fired >= cooldown:
                    timer.last_fired = now
                    timer.latched = True
                    fired = True
                duration = no_blink_dur
                # Chỉ hiển thị vào "Hành vi đang nghi ngờ" khi đã quá 30s không chớp mắt
                show_active = not closed and no_blink_dur >= 30.0
            else:
                fired, duration = timer.update(cond, now, min_dur, cooldown)
                show_active = bool(cond)

            if show_active:
                active.append({
                    "code": code,
                    "label": label,
                    "duration": round(duration, 1),
                    "required": min_dur,
                    "severity": severity,
                    "latched": timer.latched,
                })
            if fired:
                new_events.append({
                    "code": code,
                    "label": label,
                    "severity": severity,
                    "duration": round(duration, 1),
                })

        evidence_url = None
        if new_events:
            evidence_url = _save_evidence(state, frame, feats, phone_detections, new_events, metrics)
            stamp = datetime.fromtimestamp(now).isoformat(timespec="seconds")
            for ev in new_events:
                ev.update(time=stamp, evidence_image=evidence_url)
                state.counts[ev["code"]] = state.counts.get(ev["code"], 0) + 1
                state.events.append(ev)
            _write_logs(state, new_events, metrics)

        return {
            "calibration": {
                "done": state.calibrated,
                "progress": round(progress, 2),
                "avatar_url": b.avatar_url,
            },
            "metrics": metrics,
            "active_flags": active,
            "violations": new_events,
            "evidence_image": evidence_url,
            "counts": dict(state.counts),
            "total_violations": sum(state.counts.values()),
            "overlay": _overlay(feats, state.student_name),
        }


# ---------------------------------------------------------------------------
# Lớp phủ cho giao diện web
# ---------------------------------------------------------------------------
def _overlay(feats: FrameFeatures, student_name: str = "") -> dict:
    return {
        "faces": [
            {
                "bbox": f.bbox,
                "nose": f.nose_2d,
                "dir": f.nose_dir_2d,
                "irises": f.iris_points,
                "similarity": f.similarity,
                "is_registered": f.is_registered,
                "label": (
                    f"{student_name} ({f.similarity:.0f}%)"
                    if f.is_registered
                    else (f"NGƯỜI LẠ ({f.similarity:.0f}%)" if f.similarity is not None else "")
                ),
            }
            for f in feats.faces
        ],
        "poses": [{"shoulders": p.shoulders, "wrists": p.wrists} for p in feats.poses],
    }


# ---------------------------------------------------------------------------
# Ảnh bằng chứng & nhật ký
# ---------------------------------------------------------------------------
def annotate(frame: np.ndarray, feats: FrameFeatures, phones: list[dict] | None,
             title: str, lines: list[str]) -> np.ndarray:
    # Lật ngang ảnh bằng chứng để khớp 100% với chiều hiển thị trên màn hình (mirror)
    img = cv2.flip(frame, 1)
    w = img.shape[1]

    def flip_pt(pt: tuple[int, int]) -> tuple[int, int]:
        return (w - 1 - pt[0], pt[1])

    for i, f in enumerate(feats.faces):
        if f.is_registered is True:
            color = (0, 200, 80)       # Xanh lá: đúng thí sinh
        elif f.is_registered is False:
            color = (0, 0, 255)        # Đỏ: người lạ / thi hộ
        else:
            color = (0, 0, 255) if i == 0 else (0, 165, 255)
        x1, y1, x2, y2 = f.bbox
        cv2.rectangle(img, (w - 1 - x2, y1), (w - 1 - x1, y2), color, 2)
        for contour in f.eye_contours:
            pts = np.array([flip_pt(p) for p in contour], np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 1)
        for p in f.iris_points:
            cv2.circle(img, flip_pt(p), 3, (255, 255, 0), -1)
        cv2.arrowedLine(img, flip_pt(f.nose_2d), flip_pt(f.nose_dir_2d), (255, 0, 255), 2, tipLength=0.2)
    for p in feats.poses:
        cv2.line(img, flip_pt(p.shoulders[0]), flip_pt(p.shoulders[1]), (255, 128, 0), 3)
        for wr in p.wrists:
            cv2.circle(img, flip_pt(wr), 6, (255, 128, 0), -1)
    for d in phones or []:
        cv2.rectangle(img, (w - 1 - d["x2"], d["y1"]), (w - 1 - d["x1"], d["y2"]), (0, 0, 255), 3)

    # Thanh tiêu đề (hỗ trợ tiếng Việt qua Pillow nếu có font)
    bar_h = 26 + 20 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    font_path = next((p for p in _FONT_PATHS if Path(p).exists()), None) if Image else None
    if font_path:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((8, 4), title, font=ImageFont.truetype(font_path, 18), fill=(255, 80, 80))
        small = ImageFont.truetype(font_path, 14)
        for i, line in enumerate(lines):
            draw.text((8, 28 + 20 * i), line, font=small, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, title.encode("ascii", "ignore").decode(), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
        for i, line in enumerate(lines):
            cv2.putText(img, line.encode("ascii", "ignore").decode(), (8, 42 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def _save_evidence(state, frame, feats, phones, events, metrics) -> str:
    sid = safe_id(state.student_id)
    folder = config.EVIDENCE_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    codes = "-".join(e["code"] for e in events)
    filename = f"{ts}_{codes}.jpg"
    title = "VI PHẠM: " + ", ".join(e["label"] for e in events)
    info = [
        f"{state.student_name} ({state.student_id}) - {datetime.now():%d/%m/%Y %H:%M:%S}",
        "Yaw {yaw}°  Pitch {pitch}°  Mắt H {gaze_h}  V {gaze_v}  Mặt: {faces}".format(
            yaw=metrics.get("d_yaw", metrics.get("yaw", "-")),
            pitch=metrics.get("d_pitch", metrics.get("pitch", "-")),
            gaze_h=metrics.get("d_gaze_h", "-"), gaze_v=metrics.get("d_gaze_v", "-"),
            faces=metrics.get("faces")),
    ]
    img = annotate(frame, feats, phones, title, info)
    cv2.imwrite(str(folder / filename), img)
    return f"/static/evidence/{sid}/{filename}"


CSV_FIELDS = ["time", "student_id", "student_name", "code", "label", "severity",
              "duration", "evidence_image", "metrics"]


def report_paths(student_id: str) -> tuple[Path, Path]:
    folder = config.REPORTS_DIR / safe_id(student_id)
    return folder / "violations.csv", folder / "violations.jsonl"


def _write_logs(state: SessionState, events: list[dict], metrics: dict) -> None:
    csv_path, jsonl_path = report_paths(state.student_id)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for ev in events:
            writer.writerow({
                "time": ev["time"], "student_id": state.student_id, "student_name": state.student_name,
                "code": ev["code"], "label": ev["label"], "severity": ev["severity"],
                "duration": ev["duration"], "evidence_image": ev["evidence_image"],
                "metrics": json.dumps(metrics, ensure_ascii=False, default=str),
            })
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps({**ev, "student_id": state.student_id,
                                 "student_name": state.student_name, "metrics": metrics},
                                ensure_ascii=False, default=str) + "\n")


def read_report(student_id: str) -> list[dict]:
    _, jsonl_path = report_paths(student_id)
    if not jsonl_path.exists():
        return []
    with open(jsonl_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
