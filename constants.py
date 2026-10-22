"""项目级共享常量：四分类的编码、中文名与展示顺序。"""

CATEGORY_ORDER = ["recyclable", "kitchen", "hazardous", "other"]

CATEGORY_NAMES = {
    "recyclable": "可回收物",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
}

CATEGORY_EN_NAMES = {
    "recyclable": "Recyclable",
    "kitchen": "Food Waste",
    "hazardous": "Hazardous",
    "other": "Residual Waste",
}
