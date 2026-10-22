"""全局配置，全部支持环境变量覆盖，方便本机演示与服务器部署。"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"

_MODEL_PATH = Path(os.getenv("MODEL_PATH", str(WEIGHTS_DIR / "best.pt")))
MODEL_PATH = _MODEL_PATH if _MODEL_PATH.is_absolute() else BASE_DIR / _MODEL_PATH

# 单张图片上传上限（MB）
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# 未训练模型时是否允许“演示模式”兜底，保证系统雏形可直接跑通
DEMO_FALLBACK = os.getenv("DEMO_FALLBACK", "1").lower() not in {"0", "false", "no"}

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes"}
