"""
Điểm khởi chạy cho PyCharm: chuột phải file này -> Run 'run'.

- Kiểm tra thư viện cần thiết
- Tự tải model MediaPipe nếu thiếu
- Khởi động máy chủ uvicorn và tự mở trình duyệt
"""
import importlib
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)                      # Working directory luôn là thư mục dự án
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("GLOG_minloglevel", "2")       # bớt log của MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for _stream in (sys.stdout, sys.stderr):      # tránh lỗi in tiếng Việt trên console Windows
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HOST = "127.0.0.1"
PORT = int(os.environ.get("EDUGUARD_PORT", "8000"))

REQUIRED = {
    "fastapi": "fastapi", "uvicorn": "uvicorn[standard]", "jinja2": "jinja2",
    "multipart": "python-multipart", "cv2": "opencv-python", "numpy": "numpy",
    "ultralytics": "ultralytics", "mediapipe": "mediapipe", "PIL": "pillow",
}


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def check_dependencies() -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if not missing:
        return

    print("Thieu thu vien: " + ", ".join(missing) + " -> dang tu dong cai dat...")
    try:
        normal = [p for p in missing if p != "mediapipe"]
        if normal:
            _pip(*normal)
        if "mediapipe" in missing:
            # --no-deps: tránh cài opencv-contrib-python đè lên opencv-python
            _pip("absl-py", "sounddevice")
            _pip("--no-deps", "mediapipe")
    except subprocess.CalledProcessError:
        print("Cai dat that bai. Neu bao 'Access is denied', hay dung moi tien trinh Python/uvicorn dang chay roi thu lai.")
        sys.exit(1)
    importlib.invalidate_caches()


def main() -> None:
    check_dependencies()

    from backend.model_manager import ensure_models
    ensure_models()

    import uvicorn

    url = f"http://{HOST}:{PORT}"
    threading.Timer(4.0, lambda: webbrowser.open(url)).start()
    print(f"EduGuard AI dang chay tai {url}  (Ctrl+C hoac nut Stop de dung)")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
