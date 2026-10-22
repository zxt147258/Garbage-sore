"""模型推理器：优先加载训练好的权重，缺失时可选进入演示模式。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image

from backend.advice import get_advice
from config import DEMO_FALLBACK, MODEL_PATH
from constants import CATEGORY_NAMES, CATEGORY_ORDER
from model.labeling import resolve_category


class ModelLoadError(RuntimeError):
    """模型权重或运行环境不可用。"""


class Predictor:
    """线程安全的单例推理器，首次调用时惰性加载模型。"""

    def __init__(self, model_path: Path = MODEL_PATH, demo_fallback: bool = DEMO_FALLBACK):
        self.model_path = Path(model_path)
        self.demo_fallback = demo_fallback
        self._lock = threading.Lock()
        self._load_attempted = False
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._meta: dict = {}
        self._load_error: str | None = None

    # ------------------------------------------------------------------ 加载
    def load(self) -> None:
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            if not self.model_path.exists():
                self._load_error = f"未找到模型权重：{self.model_path}"
                return

            try:
                import torch

                from model.dataset import build_transforms
                from model.net import build_model
            except Exception as exc:  # 未安装 PyTorch / torchvision
                self._load_error = f"模型运行环境不可用：{exc}"
                return

            try:
                checkpoint = torch.load(
                    self.model_path, map_location="cpu", weights_only=True
                )
                class_names = checkpoint.get("class_names", CATEGORY_ORDER)
                arch = checkpoint.get("arch", "lightresnet")
                image_size = int(checkpoint.get("image_size", 224))

                model = build_model(arch, num_classes=len(class_names))
                model.load_state_dict(checkpoint["state_dict"])

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(self._device).eval()

                self._model = model
                self._transform = build_transforms(image_size, train=False)
                self._meta = {
                    "arch": arch,
                    "image_size": image_size,
                    "class_names": class_names,
                    "best_val_acc": checkpoint.get("best_val_acc"),
                    "trained_at": checkpoint.get("trained_at"),
                    "params": sum(p.numel() for p in model.parameters()),
                }
            except Exception as exc:
                self._load_error = f"模型加载失败：{exc}"
                self._model = None

    # ------------------------------------------------------------------ 状态
    @property
    def ready(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "demo_mode": not self.ready and self.demo_fallback,
            "model_path": str(self.model_path),
            "load_error": self._load_error,
            "model": self._meta,
        }

    # ------------------------------------------------------------------ 推理
    def predict(self, image: Image.Image, hint: str | None = None) -> dict:
        started = time.perf_counter()
        self.load()

        if self._model is not None:
            result = self._predict_with_model(image)
            result["demo_mode"] = False
        elif self.demo_fallback:
            result = self._predict_demo(image, hint)
            result["demo_mode"] = True
            result["demo_note"] = (
                "当前未加载训练权重，展示的是演示模式结果（基于文件名与颜色启发式）。"
                "完成训练后放入 weights/best.pt 即可切换为真实模型推理。"
            )
        else:
            raise ModelLoadError(self._load_error or "模型未就绪，请先训练模型。")

        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        result["model"] = self._meta or {"arch": "demo", "image_size": None}
        return result

    def _predict_with_model(self, image: Image.Image) -> dict:
        import torch

        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        class_names = self._meta.get("class_names", CATEGORY_ORDER)
        k = min(3, len(class_names))
        top_probs, top_idx = probs.topk(k)
        top_k = [
            {
                "category": class_names[int(idx)],
                "label": CATEGORY_NAMES.get(class_names[int(idx)], class_names[int(idx)]),
                "probability": round(float(prob), 4),
            }
            for prob, idx in zip(top_probs, top_idx)
        ]
        best = top_k[0]
        return {
            "category": best["category"],
            "label": best["label"],
            "confidence": best["probability"],
            "top_k": top_k,
            "advice": get_advice(best["category"]),
        }

    def _predict_demo(self, image: Image.Image, hint: str | None) -> dict:
        code, signal = self._demo_category(image, hint)
        confidence = round(min(0.58, 0.4 + signal * 0.18), 4)

        others = [item for item in CATEGORY_ORDER if item != code][:2]
        weights = [0.6, 0.4]
        rest = 1.0 - confidence
        top_k = [{"category": code, "label": CATEGORY_NAMES[code], "probability": confidence}]
        for other, weight in zip(others, weights):
            top_k.append(
                {
                    "category": other,
                    "label": CATEGORY_NAMES[other],
                    "probability": round(rest * weight, 4),
                }
            )
        top_k.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "category": code,
            "label": CATEGORY_NAMES[code],
            "confidence": confidence,
            "top_k": top_k,
            "advice": get_advice(code),
        }

    @staticmethod
    def _demo_category(image: Image.Image, hint: str | None) -> tuple[str, float]:
        """演示模式：先用文件名关键词，再用简单的颜色统计兜底。"""
        if hint:
            code = resolve_category(hint)
            if code:
                return code, 0.9

        small = image.convert("RGB").resize((64, 64))
        pixels = list(small.getdata())

        total = len(pixels)
        green = brown = bright = red = blue = 0
        for r, g, b in pixels:
            if g > r * 1.12 and g > b * 1.08 and g > 60:
                green += 1
            if r > 70 and r > g > b and (r - b) > 25:
                brown += 1
            if (r + g + b) / 3 > 185:
                bright += 1
            if r > 130 and r > g * 1.4 and r > b * 1.4:
                red += 1
            if b > 110 and b > r * 1.15 and b > g * 1.05:
                blue += 1

        green_ratio = green / total
        brown_ratio = brown / total
        bright_ratio = bright / total
        red_ratio = red / total
        blue_ratio = blue / total

        if green_ratio > 0.24 or brown_ratio > 0.3:
            return "kitchen", green_ratio + brown_ratio
        if red_ratio > 0.2:
            return "hazardous", red_ratio
        if blue_ratio > 0.2 or bright_ratio > 0.55:
            return "recyclable", max(blue_ratio, bright_ratio)
        return "other", 0.3
