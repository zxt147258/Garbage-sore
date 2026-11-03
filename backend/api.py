"""REST API：图片上传推理、健康检查与分类指引。"""

from __future__ import annotations

import io
import os

from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError

from backend.advice import category_catalog
from backend.predictor import ModelLoadError
from config import ALLOWED_EXTENSIONS, MAX_IMAGE_PIXELS, MAX_UPLOAD_MB

api_bp = Blueprint("api", __name__)


def _error(message: str, status: int = 400, code: str = "bad_request"):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status


def _predictor():
    return current_app.extensions["garbage_predictor"]


@api_bp.get("/health")
def health():
    predictor = _predictor()
    status = predictor.status()
    return jsonify(
        {
            "success": True,
            "data": {
                **status,
                "service": "campus-garbage-classifier",
                "max_upload_mb": MAX_UPLOAD_MB,
            },
        }
    )


@api_bp.get("/categories")
def categories():
    return jsonify({"success": True, "data": category_catalog()})


@api_bp.post("/predict")
def predict():
    upload = request.files.get("file")
    if upload is None:
        return _error("请求缺少图片文件（字段名应为 file）。")

    filename = os.path.basename(upload.filename or "")
    if not filename:
        return _error("图片文件名无效。")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return _error(
            f"不支持的图片格式 .{ext or '未知'}，请上传 JPG、PNG、WEBP 或 BMP 图片。",
            code="unsupported_type",
        )

    raw = upload.read()
    if not raw:
        return _error("图片内容为空。")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return _error(f"图片超过 {MAX_UPLOAD_MB}MB 限制。", status=413, code="file_too_large")

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            width, height = probe.size
            if width <= 0 or height <= 0:
                return _error("图片尺寸无效。", code="invalid_image")
            if width * height > MAX_IMAGE_PIXELS:
                return _error(
                    f"图片分辨率过大（{width}×{height}），请压缩到 "
                    f"{MAX_IMAGE_PIXELS // 1_000_000}MP 以内。",
                    status=413,
                    code="image_too_large",
                )
            probe.verify()
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return _error("无法解析该图片，请确认文件未损坏。", code="invalid_image")

    try:
        result = _predictor().predict(image, hint=filename)
    except ModelLoadError as exc:
        return _error(str(exc), status=503, code="model_not_ready")
    except Exception:
        current_app.logger.exception("推理失败")
        return _error("识别过程出现异常，请稍后重试。", status=500, code="inference_error")

    return jsonify({"success": True, "data": result})
