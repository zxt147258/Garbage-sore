"""校园垃圾智能分类系统入口。

启动：
    python app.py
默认地址：http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, send_from_directory

from backend.api import api_bp
from backend.predictor import Predictor
from config import DEBUG, FRONTEND_DIR, HOST, MAX_CONTENT_LENGTH, MODEL_PATH, PORT


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIR),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.json.ensure_ascii = False

    predictor = Predictor(model_path=MODEL_PATH)
    predictor.load()
    app.extensions["garbage_predictor"] = predictor

    app.register_blueprint(api_bp, url_prefix="/api")

    status = predictor.status()
    if status["ready"]:
        app.logger.info("模型已就绪：%s", status["model"].get("arch"))
    else:
        app.logger.warning("模型未就绪：%s（演示模式=%s）", status["load_error"], status["demo_mode"])

    @app.get("/")
    def index():
        response = send_from_directory(FRONTEND_DIR, "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.errorhandler(413)
    def too_large(_):
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "code": "file_too_large",
                            "message": f"图片超过 {MAX_CONTENT_LENGTH // (1024 * 1024)}MB 限制。",
                        },
                    }
                ),
                413,
            )
        return "Request entity too large", 413

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {"code": "not_found", "message": "接口不存在。"},
                    }
                ),
                404,
            )
        return "Not Found", 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {"code": "method_not_allowed", "message": "请求方法不允许。"},
                    }
                ),
                405,
            )
        return "Method Not Allowed", 405

    @app.after_request
    def add_headers(response):
        # 前后端分离时允许跨域；同域部署时该头无副作用
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host=HOST, port=PORT, debug=DEBUG)
