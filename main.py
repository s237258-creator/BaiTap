from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import numpy as np

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend import config
from backend.detector import detect_phone, get_model
from backend.behavior_analyzer import get_analyzer
from backend import behavior_tracker as tracker

BASE_DIR = Path(__file__).resolve().parent

# Tạo thư mục lưu trữ ảnh bằng chứng nếu chưa có
EVIDENCE_DIR = config.EVIDENCE_DIR
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Tải sẵn model khi khởi động để khung hình đầu tiên không bị chậm
    get_model()
    get_analyzer()
    yield


app = FastAPI(
    title="EduGuard AI",
    description="Hệ thống giám sát hành vi sử dụng thiết bị di động, tư thế, cử động đầu và hướng nhìn",
    version="2.0.0",
    lifespan=lifespan
)

# Cho phép gọi API ngay cả khi mở index.html từ Built-in Preview của PyCharm (cổng 63342)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình giao diện và file tĩnh
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)



@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/api/health")
async def health():
    return {
        "status": "success",
        "message": "EduGuard AI đang hoạt động"
    }


@app.post("/api/session/start")
def session_start(
    student_name: str = Form("Chưa rõ"),
    student_id: str = Form("Chưa rõ")
):
    """Bắt đầu phiên mới: reset trạng thái và hiệu chuẩn lại tư thế chuẩn."""
    tracker.start_session(student_id, student_name)
    return {
        "success": True,
        "calibration_seconds": config.CALIBRATION_SECONDS,
        "message": "Hãy ngồi thẳng, nhìn vào màn hình và để hai tay trên bàn trong vài giây để hiệu chuẩn."
    }


@app.post("/api/session/stop")
def session_stop(student_id: str = Form("Chưa rõ")):
    summary = tracker.end_session(student_id)
    return {"success": True, "summary": summary}


@app.post("/api/session/recalibrate")
def session_recalibrate(
    student_name: str = Form("Chưa rõ"),
    student_id: str = Form("Chưa rõ")
):
    tracker.recalibrate_session(student_id, student_name)
    return {
        "success": True,
        "calibration_seconds": config.CALIBRATION_SECONDS,
        "message": "Đang hiệu chuẩn lại tư thế chuẩn..."
    }


@app.post("/api/session/reset")
def session_reset(
    student_name: str = Form("Chưa rõ"),
    student_id: str = Form("Chưa rõ")
):
    tracker.reset_student_history(student_id, student_name)
    return {"success": True, "message": "Đã xóa sạch lịch sử vi phạm và bằng chứng của thí sinh."}


@app.get("/api/report/{student_id}")
def report_json(student_id: str):
    return {"student_id": student_id, "events": tracker.read_report(student_id)}


@app.get("/api/report/{student_id}/csv")
def report_csv(student_id: str):
    csv_path, _ = tracker.report_paths(student_id)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu vi phạm cho thí sinh này")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"bao_cao_{tracker.safe_id(student_id)}.csv"
    )


# Lưu ý: dùng "def" (không async) để FastAPI chạy trong threadpool,
# tránh việc suy luận AI nặng làm treo vòng lặp sự kiện.
@app.post("/api/detect")
def detect(
    file: UploadFile = File(...),
    student_name: str = Form("Chưa rõ"),
    student_id: str = Form("Chưa rõ"),
    check_phone: bool = Form(True)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Dữ liệu gửi lên không phải hình ảnh"
        )

    image_bytes = file.file.read()

    if len(image_bytes) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Hình ảnh vượt quá 5MB"
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Không đọc được hình ảnh"
        )

    try:
        # 1) YOLO phát hiện điện thoại (chỉ chạy khi client yêu cầu, ~1 lần/giây)
        detections = detect_phone(frame) if check_phone else None

        # 2) MediaPipe + OpenCV: tư thế, đầu, mắt, chất lượng ảnh
        features = get_analyzer().analyze(frame)

        # 3) Luật thời gian + lưu bằng chứng + nhật ký
        session = tracker.get_session(student_id, student_name)
        behavior = tracker.process_frame(session, frame, features, detections)

        return {
            "success": True,
            "phone_checked": detections is not None,
            "phone_detected": bool(detections),
            "total": len(detections or []),
            "detections": detections or [],
            **behavior,
        }

    except Exception as error:
        print("Lỗi AI:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Không thể phân tích hình ảnh"
        )
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    # Tự động mở trình duyệt sau 1 giây khi server khởi động
    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")

    threading.Timer(1.0, open_browser).start()

    # Chạy server
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)