"""
Tự động tải các model MediaPipe Tasks (.task) nếu chưa có trong thư mục models/.
"""
import urllib.request
from pathlib import Path

from backend import config


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[model] Dang tai {dest.name} ...")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    print(f"[model] Da tai xong {dest.name} ({dest.stat().st_size // 1024} KB)")


def ensure_models() -> None:
    """Đảm bảo các model cần thiết đã tồn tại; tải về nếu thiếu."""
    for url, path in (
        (config.FACE_MODEL_URL, config.FACE_MODEL_PATH),
        (config.POSE_MODEL_URL, config.POSE_MODEL_PATH),
        (config.SFACE_MODEL_URL, config.SFACE_MODEL_PATH),
    ):
        if path.exists() and path.stat().st_size > 0:
            continue
        try:
            _download(url, path)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"Khong tai duoc model {path.name}. Kiem tra ket noi mang hoac tai thu cong "
                f"tu {url} va dat vao {path}"
            ) from error


if __name__ == "__main__":
    ensure_models()

