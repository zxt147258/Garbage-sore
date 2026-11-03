# 绿识 GreenSort · 校园垃圾智能分类系统

面向校园场景的图像垃圾分类系统雏形：拍照上传垃圾图片，输出 **可回收物 / 厨余垃圾 / 有害垃圾 / 其他垃圾** 四分类结果、置信度与投放指引。

系统覆盖「数据采集 → 清洗划分 → 模型训练 → 评估调参 → Web 部署推理」的完整流程，对应《人工智能原理与实践》期末作品要求。

---

## 1. 当前完成度

| 模块 | 状态 |
| --- | --- |
| Flask 后端 + REST API | 已完成，可直接启动 |
| Web 前端（上传、预览、结果、指引） | 已完成，响应式布局 |
| 演示模式（无权重时可跑通全流程） | 已完成 |
| 数据清洗、标签归并、分层划分、5 折数据生成 | 已完成 |
| 轻量 ResNet / SimpleCNN / ResNet18 迁移学习 | 已完成 |
| 训练、测试集评估、网格搜索调参 | 已完成 |
| 真实模型权重 | 待使用公开数据集训练生成 |

> 首次运行如果 `weights/best.pt` 不存在，系统会进入**演示模式**：用文件名关键词 + 简单颜色统计给出结果，仅用于演示交互流程，页面会明确标注。训练完成后放入 `weights/best.pt` 即自动切换真实模型。

---

## 2. 系统架构

```text
┌────────────────────── 浏览器（前端） ──────────────────────┐
│  index.html + style.css + app.js                          │
│  拖拽上传 / 图片预览 / 结果卡片 / 分类指南 / 服务状态       │
└───────────────────────────┬───────────────────────────────┘
                            │ POST /api/predict (multipart)
┌───────────────────────────▼───────────────────────────────┐
│ Flask (app.py)                                            │
│  ├─ /api/health       服务与模型状态                       │
│  ├─ /api/categories   四分类与投放指引                     │
│  └─ /api/predict      校验图片 → 预处理 → 推理 → 组装结果   │
│         │                                                 │
│         ▼                                                 │
│  Predictor（懒加载、线程安全）                              │
│         │                                                 │
│         ▼                                                 │
│  LightResNet / SimpleCNN / ResNet18                       │
│  weights/best.pt                                          │
└───────────────────────────┬───────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ 离线训练流水线             │
              │ prepare_data → train      │
              │ → evaluate / tune         │
              └───────────────────────────┘
```

前后端通过 JSON API 解耦，前端也可单独部署（API 地址通过 `window.API_BASE` 配置），满足“前后端分离”加分项。

---

## 3. 目录结构

```text
垃圾智能分类/
├── app.py                    # Flask 入口，静态页面 + API
├── config.py                 # 路径、上传限制、演示模式等配置
├── constants.py              # 四分类编码与中文名
├── requirements.txt
├── backend/
│   ├── api.py                # 上传校验、健康检查、分类目录接口
│   ├── predictor.py          # 权重加载、真实推理、演示模式兜底
│   └── advice.py             # 四分类投放指引数据
├── model/
│   ├── labeling.py           # 子类别 → 四分类的别名/关键词规则
│   ├── prepare_data.py       # 清洗、去重、分层划分、K 折
│   ├── dataset.py            # Dataset / 数据增强 / DataLoader
│   ├── net.py                # LightResNet、SimpleCNN、build_model
│   ├── train.py              # 训练、验证、早停、权重与指标保存
│   ├── evaluate.py           # 测试集指标与混淆矩阵
│   └── tune.py               # 学习率/权重衰减网格搜索
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── data/
│   ├── label_mapping.json    # 可选：显式子类别映射
│   ├── raw/                  # 原始数据集（需自行下载，已 gitignore）
│   └── splits/               # 生成的 CSV 划分（已 gitignore）
└── weights/                  # best.pt 与训练元数据（已 gitignore）
```

---

## 4. 快速开始

### 4.1 环境

当前工作区已创建 `.venv`，并验证 **Python 3.14.6 + PyTorch 2.14.1（CPU）+ torchvision 0.29.1** 可正常训练与推理，启动前只需激活：

```powershell
cd d:\垃圾智能分类
.\.venv\Scripts\Activate.ps1
```

如需在新机器复现，建议无 NVIDIA 显卡时安装 CPU 版 PyTorch（体积更小）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

若其他 Python 版本安装 PyTorch 失败，可改用 **Python 3.11 / 3.12**。

### 4.2 启动系统

```powershell
python app.py
```

浏览器打开 <http://127.0.0.1:5000>。

可选环境变量：

```powershell
$env:PORT="8000"
$env:MODEL_PATH="weights/best.pt"
$env:DEMO_FALLBACK="0"     # 关闭演示模式，模型缺失时直接返回 503
$env:MAX_UPLOAD_MB="8"
$env:MAX_IMAGE_PIXELS="40000000"   # 单图最大解码像素，防止小文件解压成超大图
python app.py
```

---

## 5. 数据集与预处理

### 5.1 数据来源（公开、合法合规）

1. **PaddleX / 华为云 40 类中文垃圾分类数据集**：40 个中文子类别，覆盖四分类，适合校园与国内投放标准。
2. **Kaggle 12 类生活垃圾数据集**：纸、纸板、玻璃、金属、塑料、织物、电池、厨余、其他等，适合作为基础数据。
3. **其他可选公开数据集**：RealWaste、TACO 等，可继续补充真实场景样本。
4. 校园自采数据：在食堂、宿舍、教学楼拍摄并人工标注，注意避开人脸、隐私信息，取得拍摄许可。

> **当前已下载 107,069 张原始图片**，来源如下（均已归并到四分类）：
>
> | 数据集 | 使用张数 | 说明 |
> | --- | ---: | --- |
> | [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | 15,515 | 12 个子类，基础数据 |
> | [PaddleX trash40](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/trash40.tar) | 16,542 | 40 个中文子类，贴合国内标准 |
> | [Recyclable and Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) | 15,000 | 30 个子类，MIT |
> | [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data) | 25,077 | 厨余 / 可回收，CC BY-SA 4.0 |
> | [Garbage Dataset Classification](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification) | 13,901 | 6 个材料类别 |
> | [Garbage Classification Dataset](https://www.kaggle.com/datasets/namanjain001/garbage-classification-dataset) | 8,775 | 10 个子类 |
> | [Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) | 12,259 | 10 个子类，MIT |
>
> 清洗时扫描 **107,069 张**，按 MD5 去重 16,761 张、剔除过小图片 7 张，最终保留 **90,301 张唯一图片**：
> 可回收物 59,162 / 厨余垃圾 20,087 / 有害垃圾 3,225 / 其他垃圾 7,827。
>
> 已按 70% / 15% / 15% 分层划分：train 63,207 / val 13,547 / test 13,547，并生成 5 折交叉验证数据。类别分布不均衡，训练脚本默认启用类别权重。
>
> 所有数据仅用于课程研究，请遵守各数据集原始许可并在报告中注明来源。原始数据统一放在 `data/raw/`。

### 5.2 支持的目录形式

```text
# 形式 A：子类别目录（推荐）
data/raw/
├── battery/            # 自动映射为有害垃圾
├── biological/         # 厨余垃圾
├── cardboard/          # 可回收物
├── glass/
├── paper/
├── plastic/
├── metal/
└── trash/              # 其他垃圾

# 形式 B：已经是四分类目录
data/raw/
├── recyclable/
├── kitchen/
├── hazardous/
└── other/

# 形式 C：华为云数字目录 + garbage_label.txt
data/raw/garbage_classification/
├── 0/
├── 1/
├── ...
└── garbage_label.txt
```

中文文件夹名（`纸箱`、`果皮`、`过期药品` 等）同样适用。无法识别的目录会被跳过并在终端汇总；可用 `data/label_mapping.json` 显式指定：

```json
{
  "label_map": {
    "0": "kitchen",
    "1": "other",
    "my-custom-label": "hazardous"
  }
}
```

### 5.3 生成训练 / 验证 / 测试集

```powershell
python -m model.prepare_data --raw data/raw --out data/splits --cv-folds 5
```

脚本会完成：

- 遍历图片，按子类别分层抽样，默认 70% / 15% / 15%；
- 校验图片完整性、过滤过小图片、MD5 去重；
- 输出 `train.csv`、`val.csv`、`test.csv` 与 `summary.json`；
- 可选生成 `data/splits/cv/fold_1..5/` 五折交叉验证数据。

查看 `data/splits/summary.json` 可确认四分类样本量与跳过原因。

---

## 6. 模型训练

### 6.1 轻量 ResNet（默认，适合 CPU 演示）

```powershell
python -m model.train --arch lightresnet --epochs 30 --batch-size 32 --lr 0.001 --patience 6
```

### 6.2 简单 CNN 基线

```powershell
python -m model.train --arch simplecnn --epochs 30 --lr 0.001 --out weights/simplecnn
```

### 6.3 ResNet18 迁移学习（需要联网下载 ImageNet 权重）

```powershell
python -m model.train --arch resnet18 --pretrained --epochs 15 --batch-size 16 --lr 0.0003
```

训练特性：数据增强、类别权重缓解不平衡、Label Smoothing、AdamW、余弦退火、验证集早停。

输出（默认 `weights/`）：

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 最佳验证集权重 + 类别顺序 + 图像尺寸 |
| `model_meta.json` | 模型结构、参数量、最佳轮次、样本量 |
| `history.json` | 每轮 loss / accuracy / macro-F1 |
| `metrics.json` | 最佳轮次的分类指标与混淆矩阵 |

> 训练完成后保持 `weights/best.pt` 路径不变，刷新页面即可切换真实模型，无需改代码。

### 6.4 测试集评估

```powershell
python -m model.evaluate --checkpoint weights/best.pt --test-csv data/splits/test.csv
```

输出准确率、宏平均 F1、各类别 precision / recall / F1 与混淆矩阵，并保存 `weights/eval_report.json`。

### 6.5 超参数调优

```powershell
python -m model.tune --epochs 8 --archs lightresnet simplecnn `
  --lrs 0.001 0.0005 --weight-decays 0.0001 0.0005
```

结果保存在 `weights/tune/tune_results.csv`，按验证集准确率排序，可直接放入报告。

### 6.6 交叉验证

```powershell
python -m model.train --fold 1 --epochs 15
python -m model.train --fold 2 --epochs 15
# ... 依次运行 fold 3-5，汇总各折 best_val_acc
```

---

## 7. API 说明

### GET `/api/health`

```json
{
  "success": true,
  "data": {
    "ready": true,
    "demo_mode": false,
    "model_path": "D:\\垃圾智能分类\\weights\\best.pt",
    "load_error": null,
    "model": { "arch": "lightresnet", "image_size": 224, "params": 1560000 }
  }
}
```

### POST `/api/predict`

`multipart/form-data`，字段名 `file`，支持 JPG / PNG / WEBP / BMP，默认文件上限 8MB、解码分辨率上限 40MP，均可通过环境变量调整。

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/predict -F "file=@D:\photos\bottle.jpg"
```

```json
{
  "success": true,
  "data": {
    "category": "recyclable",
    "label": "可回收物",
    "confidence": 0.9231,
    "top_k": [
      { "category": "recyclable", "label": "可回收物", "probability": 0.9231 },
      { "category": "other", "label": "其他垃圾", "probability": 0.0412 }
    ],
    "advice": {
      "bin_name": "蓝色可回收物桶",
      "bin_color": "#2563eb",
      "summary": "……",
      "items": ["废纸箱", "塑料瓶"],
      "tips": ["倒空瓶内液体……"],
      "campus": "教学楼、宿舍楼、快递驿站均设有蓝色回收桶。"
    },
    "demo_mode": false,
    "latency_ms": 46,
    "model": { "arch": "lightresnet", "image_size": 224 }
  }
}
```

错误统一返回：

```json
{ "success": false, "error": { "code": "invalid_image", "message": "无法解析该图片……" } }
```

常见错误码：`bad_request`、`unsupported_type`、`invalid_image`、`file_too_large`、`image_too_large`（均为 400/413）、`model_unavailable`（503）、`not_found` / `method_not_allowed`（404/405）。

---

## 8. 作业要求对照

| 课程要求 | 本项目实现 |
| --- | --- |
| 数据采集、合法合规 | 公开数据集 + 校园自采方案，注明来源与许可 |
| 数据标注与预处理 | 标签归并、图片校验、去重、分层划分、增强 |
| 模型选择与训练 | LightResNet、SimpleCNN、ResNet18 迁移学习 |
| 模型优化 | 类别权重、Label Smoothing、余弦退火、早停、网格搜索、5 折 |
| 模型部署 | Flask API 在线推理 + 前端展示，支持 CPU/GPU |
| 前后端分离 | 静态前端 + JSON API，可独立部署 |
| 系统文档 | 本 README 覆盖需求、设计、运行与接口 |

---

## 9. 后续可扩展

- 接入摄像头实时识别、批量图片上传；
- 增加 Grad-CAM 热力图，解释模型关注区域（答辩加分）；
- 使用 ONNX / TorchScript 导出，降低部署依赖；
- 增加识别历史、用户反馈纠错与数据回流；
- Docker 化部署，补充单元测试与接口测试；
- 扩充校园自采数据，按「场景 + 类别」做误差分析。

---

## 10. 已知限制

- 演示模式使用颜色与文件名启发式，**不具备真实分类能力**，报告中不可将其作为模型指标；
- 公开数据集与校园真实分布存在域差异，建议用校园自采图片做一次小规模微调；
- 四分类由子类别规则归并，个别边界样本（如污损塑料、大骨头）可能标注不一致；
- 训练权重体积较大，提交时请按课程要求单独打包，不提交到 Git。
