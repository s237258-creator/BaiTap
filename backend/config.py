"""
Cấu hình tập trung cho module AI nhận diện hành vi & gian lận.

Mọi ngưỡng (threshold) đều có thể chỉnh tại đây mà không cần sửa thuật toán.
Đơn vị thời gian: giây. Đơn vị góc: độ.
Quy ước hướng Trái / Phải: theo hướng hiển thị trên màn hình camera (mirror)
    - Trái = phía bên trái màn hình của bạn
    - Phải = phía bên phải màn hình của bạn
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
EVIDENCE_DIR = BASE_DIR / "static" / "evidence"
REPORTS_DIR = BASE_DIR / "reports"
YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Model MediaPipe Tasks (tự tải về bởi backend/model_manager.py nếu chưa có)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

# Model nhận diện khuôn mặt SFace (OpenCV Zoo ONNX) để xác minh danh tính thí sinh
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Ngưỡng xác minh danh tính khuôn mặt (Cosine similarity của SFace: chuẩn OpenCV là 0.363)
IDENTITY_COSINE_THRESHOLD = 0.38   # >= 0.38 là khớp danh tính thí sinh
IDENTITY_MAX_CHECK_YAW = 32.0      # không báo sai danh tính khi mặt đang quay quá nghiêng
IDENTITY_MAX_CHECK_PITCH = 25.0

# ---------------------------------------------------------------------------
# Hiệu chuẩn (calibration)
# ---------------------------------------------------------------------------
CALIBRATION_SECONDS = 3.0          # thời gian thí sinh nhìn thẳng màn hình
CALIBRATION_MIN_SAMPLES = 6        # số khung hình hợp lệ tối thiểu
CALIBRATION_MAX_ABS_YAW = 22.0     # mặt phải tương đối thẳng mới nhận mẫu
CALIBRATION_MAX_ABS_PITCH = 22.0

# ---------------------------------------------------------------------------
# Làm mượt tín hiệu (EMA) - giảm rung landmark
# ---------------------------------------------------------------------------
SMOOTHING_ALPHA = 0.5              # 0..1, càng lớn càng bám sát khung hình mới

# ---------------------------------------------------------------------------
# Ngưỡng hành vi (so với mốc hiệu chuẩn)
# ---------------------------------------------------------------------------
HEAD_YAW_THRESHOLD = 20.0          # độ quay đầu trái/phải (theo ma trận 3D MediaPipe)
HEAD_PITCH_DOWN_THRESHOLD = 12.0   # độ cúi đầu xuống nhìn tài liệu

# Liếc mắt ngang: kết hợp blendshape mắt + tỉ lệ mống mắt
GAZE_SCORE_H_THRESHOLD = 0.22      # độ lệch blendshape mắt sang trái/phải
GAZE_IRIS_H_THRESHOLD = 0.085      # hoặc lệch vị trí mống mắt ngang
GAZE_DOWN_SCORE_THRESHOLD = 0.28   # tăng eyeLookDown so với mốc
GAZE_IRIS_V_DOWN_THRESHOLD = 0.10  # kết hợp mống mắt hạ thấp
GAZE_MAX_HEAD_YAW = 14.0           # chỉ xét liếc mắt khi đầu gần như thẳng
GAZE_MAX_HEAD_PITCH = 10.0

# Ảnh / camera
DARK_BRIGHTNESS = 40.0             # độ sáng trung bình (0..255)
LOW_CONTRAST_STD = 12.0            # độ lệch chuẩn thấp -> camera bị che
FROZEN_PIXEL_DIFF = 1.0            # sai khác điểm ảnh trung bình giữa 2 khung
FROZEN_LANDMARK_DIFF = 0.25        # dịch chuyển landmark trung bình (pixel)
BLINK_EAR_RATIO = 0.72             # EAR < 72% EAR mốc -> đang nhắm/chớp mắt
BLINK_BLENDSHAPE = 0.35            # hoặc blendshape eyeBlink > 0.35 (bắt nhịp chớp mắt nhanh ở ~4fps)

# Thời gian trở lại bình thường tối thiểu để mở khoá báo lại cùng 1 lỗi (one-shot latch)
CLEAR_RELATCH_SECONDS = 0.8

# ---------------------------------------------------------------------------
# Danh sách các lỗi vi phạm chính (đã lược bỏ các lỗi phụ gây nhiễu):
#   code : (Nhãn hiển thị tiếng Việt, thời gian kéo dài tối thiểu (s), cooldown (s), mức độ)
# ---------------------------------------------------------------------------
RULES = {
    "PHONE":           ("Sử dụng điện thoại",                      0.8, 3.0, "high"),
    "WRONG_PERSON":    ("Sai danh tính / Nghi vấn thi hộ",         1.8, 4.0, "high"),
    "MULTIPLE_PEOPLE": ("Có nhiều người trong khung hình",         1.5, 4.0, "high"),
    "NO_FACE":         ("Rời khỏi khung hình / không thấy mặt",    2.5, 4.0, "medium"),
    "HEAD_LEFT":       ("Quay đầu sang trái",                      1.8, 3.0, "medium"),
    "HEAD_RIGHT":      ("Quay đầu sang phải",                      1.8, 3.0, "medium"),
    "HEAD_DOWN":       ("Cúi đầu xuống (nhìn tài liệu)",           2.0, 3.0, "medium"),
    "GAZE_LEFT":       ("Liếc mắt sang trái",                      1.8, 3.0, "medium"),
    "GAZE_RIGHT":      ("Liếc mắt sang phải",                      1.8, 3.0, "medium"),
    "GAZE_DOWN":       ("Liếc mắt xuống dưới",                     2.2, 3.0, "medium"),
    "CAMERA_BLOCKED":  ("Camera bị che / quá tối",                 2.0, 5.0, "high"),
    "FROZEN_FRAME":    ("Khung hình đứng yên (ảnh tĩnh/video giả)", 5.0, 8.0, "high"),
    "NO_BLINK":        ("Không chớp mắt quá 45s (nghi ảnh giả)",   45.0, 15.0, "high"),
}

# Nếu 2 khung hình cách nhau quá khoảng này, bộ đếm thời gian của hành vi bị reset
MAX_FRAME_GAP = 2.5

# Giới hạn ảnh tải lên
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
