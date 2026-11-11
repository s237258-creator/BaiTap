"""
Trích xuất đặc trưng hành vi trên MỘT khung hình bằng MediaPipe + OpenCV.

- Face Landmarker (478 điểm, có mống mắt + ma trận biến đổi 3D chuẩn của MediaPipe):
  * Đếm khuôn mặt
  * Ước lượng tư thế đầu (yaw/pitch/roll) trực tiếp từ facial_transformation_matrixes
    của MediaPipe (chính xác và không bị trôi như solvePnP 6 điểm).
  * Hướng nhìn của mắt: kết hợp vị trí mống mắt và các blendshape eyeLookIn/Out/Down.
  * Quy ước chiều Trái / Phải khớp trực tiếp với màn hình hiển thị (mirror):
      - yaw > 0  : quay đầu sang PHẢI màn hình
      - yaw < 0  : quay đầu sang TRÁI màn hình
      - pitch > 0: cúi đầu xuống
      - pitch < 0: ngửa đầu lên
- Pose Landmarker: đếm người, vẽ khung vai/tay.
- OpenCV: độ sáng, độ tương phản, độ nét (Laplacian), sai khác giữa các khung hình.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from backend import config
from backend.model_manager import ensure_models

# ---------------------------------------------------------------------------
# Chỉ số landmark của MediaPipe Face Mesh
# ---------------------------------------------------------------------------
NOSE_TIP, CHIN = 1, 152
EYE_OUTER_A, EYE_OUTER_B = 33, 263
MOUTH_A, MOUTH_B = 61, 291

# 6 điểm quanh mỗi mắt để tính EAR: [góc1, trên1, trên2, góc2, dưới2, dưới1]
EYE_A_EAR = [33, 160, 158, 133, 153, 144]
EYE_B_EAR = [362, 385, 387, 263, 373, 380]
EYE_A_LIDS = (159, 145)
EYE_B_LIDS = (386, 374)
IRIS_A = list(range(468, 473))
IRIS_B = list(range(473, 478))

# Điểm vẽ viền mắt cho ảnh bằng chứng
EYE_A_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# Pose landmarks
P_NOSE, P_L_SHOULDER, P_R_SHOULDER, P_L_WRIST, P_R_WRIST = 0, 11, 12, 15, 16


@dataclass
class FaceFeatures:
    bbox: tuple[int, int, int, int]
    yaw: float                # >0: quay sang PHẢI màn hình, <0: sang TRÁI màn hình
    pitch: float              # >0: cúi đầu xuống, <0: ngửa lên
    roll: float
    gaze_h: float             # tỉ lệ mống mắt (0..1): >0.5 lệch sang TRÁI màn hình, <0.5 lệch sang PHẢI màn hình
    gaze_v: float             # tỉ lệ mống mắt dọc (0..1): lớn = nhìn xuống
    gaze_score_h: float       # blendshape mắt ngang (-1..1): >0 nhìn sang PHẢI màn hình, <0 sang TRÁI màn hình
    ear: float
    blink_score: float        # blendshape eyeBlink trung bình (0..1)
    look_down_score: float    # blendshape eyeLookDown trung bình (0..1)
    nose_2d: tuple[int, int]
    nose_dir_2d: tuple[int, int]
    iris_points: list[tuple[int, int]]
    eye_contours: list[list[tuple[int, int]]]
    landmarks_px: np.ndarray  # (N, 2) để đo độ "đứng yên" của khuôn mặt
    area: float
    embedding: np.ndarray | None = None
    similarity: float | None = None
    is_registered: bool | None = None


@dataclass
class PoseFeatures:
    shoulders: tuple[tuple[int, int], tuple[int, int]]
    shoulder_angle: float     # độ nghiêng đường nối 2 vai
    shoulder_mid_x: float     # pixel
    shoulder_width: float     # pixel
    nose_offset: float        # (mũi - giữa vai) / bề rộng vai
    hands_visible: bool
    wrists: list[tuple[int, int]]


@dataclass
class FrameFeatures:
    width: int
    height: int
    faces: list[FaceFeatures] = field(default_factory=list)
    poses: list[PoseFeatures] = field(default_factory=list)
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    small_gray: np.ndarray | None = None

    @property
    def main_face(self) -> FaceFeatures | None:
        return max(self.faces, key=lambda f: f.area) if self.faces else None

    @property
    def main_pose(self) -> PoseFeatures | None:
        return max(self.poses, key=lambda p: p.shoulder_width) if self.poses else None


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class BehaviorAnalyzer:
    """Bao bọc các model MediaPipe; an toàn khi gọi từ nhiều luồng (dùng lock)."""

    def __init__(self) -> None:
        ensure_models()
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_buffer=config.FACE_MODEL_PATH.read_bytes()
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_buffer=config.POSE_MODEL_PATH.read_bytes()
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._face = vision.FaceLandmarker.create_from_options(face_options)
        self._pose = vision.PoseLandmarker.create_from_options(pose_options)
        self._sface = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL_PATH), "")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _extract_embedding(self, frame_bgr: np.ndarray, face: FaceFeatures) -> np.ndarray | None:
        try:
            pts = face.landmarks_px
            ir1, ir2 = face.iris_points
            if ir1[0] > ir2[0]:
                ir1, ir2 = ir2, ir1
            nose = face.nose_2d
            m1, m2 = pts[MOUTH_A], pts[MOUTH_B]
            if m1[0] > m2[0]:
                m1, m2 = m2, m1
            x1, y1, x2, y2 = face.bbox
            face_box = np.array([
                x1, y1, max(1, x2 - x1), max(1, y2 - y1),
                ir1[0], ir1[1], ir2[0], ir2[1],
                nose[0], nose[1],
                m1[0], m1[1], m2[0], m2[1], 0.99
            ], dtype=np.float32)
            aligned = self._sface.alignCrop(frame_bgr, face_box)
            feat = self._sface.feature(aligned).flatten().astype(np.float32)
            norm = float(np.linalg.norm(feat))
            return (feat / norm) if norm > 1e-6 else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    def analyze(self, frame_bgr: np.ndarray) -> FrameFeatures:
        h, w = frame_bgr.shape[:2]
        feats = FrameFeatures(width=w, height=h)

        # ---- Chất lượng ảnh (OpenCV) ----
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        feats.brightness = float(gray.mean())
        feats.contrast = float(gray.std())
        feats.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        feats.small_gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)

        # ---- MediaPipe + SFace ----
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with self._lock:
            face_result = self._face.detect(mp_image)
            pose_result = self._pose.detect(mp_image)

            blendshapes = face_result.face_blendshapes or []
            matrices = face_result.facial_transformation_matrixes or []
            for i, lms in enumerate(face_result.face_landmarks or []):
                bs = blendshapes[i] if i < len(blendshapes) else None
                mat = matrices[i] if i < len(matrices) else None
                face = self._face_features(lms, bs, mat, w, h)
                if face is not None:
                    face.embedding = self._extract_embedding(frame_bgr, face)
                    feats.faces.append(face)

        for lms in pose_result.pose_landmarks or []:
            pose = self._pose_features(lms, w, h)
            if pose is not None:
                feats.poses.append(pose)

        return feats

    # ------------------------------------------------------------------
    @staticmethod
    def _face_features(lms, blendshapes, matrix, w: int, h: int) -> FaceFeatures | None:
        pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float64)
        if len(pts) < 478:
            return None

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis(0)) if False else pts.max(axis=0)
        bbox = (int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2)))
        area = float((x2 - x1) * (y2 - y1))

        # ---- Tư thế đầu 3D từ ma trận biến đổi của MediaPipe ----
        if matrix is not None:
            rot = np.array(matrix, dtype=np.float64)[:3, :3]
            # Vector pháp tuyến hướng thẳng ra phía trước mặt trong hệ toạ độ MediaPipe:
            # +X: bên phải ảnh gốc (= bên TRÁI màn hình mirror)
            # +Y: hướng lên trên
            # +Z: hướng về phía camera
            fwd = rot @ np.array([0.0, 0.0, 1.0])
            # Đổi dấu để yaw > 0 tương ứng với quay sang PHẢI màn hình mirror (tức -X của ảnh gốc)
            yaw = math.degrees(math.atan2(-fwd[0], fwd[2]))
            # pitch > 0 tương ứng với cúi đầu xuống (-Y)
            pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))
        else:
            # Dự phòng bằng tỉ lệ hình học nếu không có ma trận
            eye_mid = (pts[EYE_OUTER_A] + pts[EYE_OUTER_B]) / 2.0
            eye_w = max(_dist(pts[EYE_OUTER_A], pts[EYE_OUTER_B]), 1.0)
            face_h = max(pts[CHIN][1] - eye_mid[1], 1.0)
            yaw = float(-(pts[NOSE_TIP][0] - eye_mid[0]) / eye_w * 90.0)
            pitch = float(((pts[NOSE_TIP][1] - eye_mid[1]) / face_h - 0.36) * 140.0)

        eye_vec = pts[EYE_OUTER_B] - pts[EYE_OUTER_A]
        roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

        nose = pts[NOSE_TIP]
        # Vẽ mũi tên chỉ hướng mặt trên ảnh (yaw > 0 = bên phải màn hình = bên trái ảnh gốc -X)
        arrow_len = max(45.0, min(w, h) * 0.14)
        dx = -math.sin(math.radians(yaw)) * arrow_len
        dy = math.sin(math.radians(pitch)) * arrow_len
        tip = np.array([nose[0] + dx, nose[1] + dy])

        # ---- Hướng nhìn của mắt (vị trí mống mắt + blendshape) ----
        def eye_ratio(corner_idx, lids, iris_idx):
            c1, c2 = pts[corner_idx[0]], pts[corner_idx[3]]
            left, right = (c1, c2) if c1[0] < c2[0] else (c2, c1)
            iris = pts[iris_idx].mean(axis=0)
            axis = right - left
            length = float(np.dot(axis, axis)) or 1.0
            ratio_h = float(np.dot(iris - left, axis) / length)
            top, bottom = pts[lids[0]], pts[lids[1]]
            span = bottom[1] - top[1]
            ratio_v = float((iris[1] - top[1]) / span) if abs(span) > 1e-3 else 0.5
            return ratio_h, ratio_v, iris

        def ear(idx):
            p = pts[idx]
            horiz = _dist(p[0], p[3]) or 1.0
            return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horiz)

        ha, va, iris_a = eye_ratio(EYE_A_EAR, EYE_A_LIDS, IRIS_A)
        hb, vb, iris_b = eye_ratio(EYE_B_EAR, EYE_B_LIDS, IRIS_B)
        ear_value = (ear(EYE_A_EAR) + ear(EYE_B_EAR)) / 2.0

        blink = look_down = gaze_score_h = 0.0
        if blendshapes:
            scores = {c.category_name: c.score for c in blendshapes}
            blink = (scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) / 2.0
            look_down = (scores.get("eyeLookDownLeft", 0.0) + scores.get("eyeLookDownRight", 0.0)) / 2.0
            # Khi mắt nhìn sang PHẢI màn hình (= trái ảnh gốc):
            # mắt phải của người (bên trái ảnh) nhìn ra ngoài (eyeLookOutRight),
            # mắt trái của người (bên phải ảnh) nhìn vào trong (eyeLookInLeft)
            look_screen_right = (scores.get("eyeLookOutRight", 0.0) + scores.get("eyeLookInLeft", 0.0)) / 2.0
            look_screen_left = (scores.get("eyeLookInRight", 0.0) + scores.get("eyeLookOutLeft", 0.0)) / 2.0
            gaze_score_h = look_screen_right - look_screen_left

        def to_int(p):
            return int(round(p[0])), int(round(p[1]))

        return FaceFeatures(
            bbox=bbox,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            gaze_h=(ha + hb) / 2.0,
            gaze_v=float(np.clip((va + vb) / 2.0, -0.5, 1.5)),
            gaze_score_h=gaze_score_h,
            ear=ear_value,
            blink_score=blink,
            look_down_score=look_down,
            nose_2d=to_int(nose),
            nose_dir_2d=to_int(tip),
            iris_points=[to_int(iris_a), to_int(iris_b)],
            eye_contours=[[to_int(pts[i]) for i in EYE_A_CONTOUR],
                          [to_int(pts[i]) for i in EYE_B_CONTOUR]],
            landmarks_px=pts[:468],
            area=area,
        )

    @staticmethod
    def _pose_features(lms, w: int, h: int) -> PoseFeatures | None:
        ls, rs = lms[P_L_SHOULDER], lms[P_R_SHOULDER]
        if min(ls.visibility or 0, rs.visibility or 0) < 0.5:
            return None
        a = np.array([ls.x * w, ls.y * h])
        b = np.array([rs.x * w, rs.y * h])
        left, right = (a, b) if a[0] < b[0] else (b, a)
        width = _dist(left, right)
        if width < 10:
            return None
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        mid = (left + right) / 2.0
        nose = lms[P_NOSE]
        nose_offset = float((nose.x * w - mid[0]) / width)

        wrists, visible = [], False
        for idx in (P_L_WRIST, P_R_WRIST):
            lm = lms[idx]
            in_frame = 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0
            if (lm.visibility or 0) >= 0.4 and in_frame:
                visible = True
                wrists.append((int(lm.x * w), int(lm.y * h)))

        return PoseFeatures(
            shoulders=((int(left[0]), int(left[1])), (int(right[0]), int(right[1]))),
            shoulder_angle=angle,
            shoulder_mid_x=float(mid[0]),
            shoulder_width=width,
            nose_offset=nose_offset,
            hands_visible=visible,
            wrists=wrists,
        )


# ---------------------------------------------------------------------------
# Singleton (tải model một lần)
# ---------------------------------------------------------------------------
_analyzer: BehaviorAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_analyzer() -> BehaviorAnalyzer:
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                print("Dang tai mo hinh MediaPipe (Face + Pose)...")
                _analyzer = BehaviorAnalyzer()
                print("Da tai MediaPipe thanh cong!")
    return _analyzer
