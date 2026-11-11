import threading

from ultralytics import YOLO

from backend.config import YOLO_MODEL_PATH

# Khởi tạo mô hình một lần
model = None
_lock = threading.Lock()


def get_model():
    global model

    if model is None:
        print("Dang tai mo hinh YOLO...")
        # Dùng đường dẫn tuyệt đối để chạy đúng dù Working Directory của PyCharm khác
        model = YOLO(str(YOLO_MODEL_PATH))
        print("Da tai mo hinh thanh cong!")

    return model


def detect_phone(frame):
    ai_model = get_model()

    with _lock:
        results = ai_model.predict(
            source=frame,
            classes=[67],
            conf=0.35,
            imgsz=640,
            verbose=False
        )

    detections = []

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            detections.append({
                "label": "Dien thoai",
                "confidence": round(confidence, 2),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

    return detections
