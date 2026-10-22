"""四分类投放指引数据，供前端结果卡片与 /api/categories 使用。"""

from constants import CATEGORY_EN_NAMES, CATEGORY_NAMES, CATEGORY_ORDER

CATEGORY_ADVICE = {
    "recyclable": {
        "bin_name": "蓝色可回收物桶",
        "bin_color": "#2563eb",
        "summary": "适宜回收和资源化利用的垃圾，投放前请保持清洁干燥。",
        "items": ["废纸箱", "塑料瓶", "易拉罐", "玻璃瓶", "旧衣物", "旧书本"],
        "tips": [
            "倒空瓶内液体，简单冲洗后压扁投放。",
            "纸箱拆平、纸张保持干燥，避免被油污污染。",
            "玻璃制品请轻放，破损玻璃需包裹后投放。",
        ],
        "campus": "教学楼、宿舍楼、快递驿站均设有蓝色回收桶。",
    },
    "kitchen": {
        "bin_name": "绿色厨余垃圾桶",
        "bin_color": "#16a34a",
        "summary": "易腐烂的生物质生活废弃物，需沥干水分后投放。",
        "items": ["剩饭剩菜", "果皮果核", "蛋壳", "茶叶渣", "菜叶", "鱼骨"],
        "tips": [
            "去除塑料袋、餐盒等包装物，只投放食物残渣。",
            "尽量沥干水分，避免污染其他垃圾。",
            "大骨头、贝壳等难降解物请投入其他垃圾。",
        ],
        "campus": "食堂收餐处设有厨余专用回收点。",
    },
    "hazardous": {
        "bin_name": "红色有害垃圾桶",
        "bin_color": "#dc2626",
        "summary": "对人体健康或自然环境造成直接或潜在危害的废弃物。",
        "items": ["废电池", "过期药品", "废灯管", "油漆桶", "杀虫剂", "水银温度计"],
        "tips": [
            "电池、灯管请单独密封，避免破损泄漏。",
            "过期药品连同包装一起投放至有害垃圾收集点。",
            "严禁混入厨余或其他垃圾，切勿自行拆解。",
        ],
        "campus": "实验楼、宿舍楼下设有红色有害垃圾收集箱。",
    },
    "other": {
        "bin_name": "灰色其他垃圾桶",
        "bin_color": "#64748b",
        "summary": "除可回收物、厨余垃圾、有害垃圾以外的其他生活废弃物。",
        "items": ["纸巾", "烟头", "陶瓷碎片", "口罩", "一次性餐具", "保鲜膜"],
        "tips": [
            "纸巾、湿巾等污染纸张请投入其他垃圾。",
            "破碎陶瓷请包裹后投放，避免划伤保洁人员。",
            "口罩请对折扎紧后投放。",
        ],
        "campus": "教室、走廊与公共区域均设有灰色其他垃圾桶。",
    },
}

# 常见别名，用于演示模式的图片文件名提示
CATEGORY_HINTS = {
    "recyclable": ["可回收", "塑料瓶", "纸箱", "易拉罐", "玻璃瓶", "旧衣服", "recyclable", "bottle", "paper"],
    "kitchen": ["厨余", "果皮", "剩饭", "菜叶", "蛋壳", "kitchen", "food", "fruit"],
    "hazardous": ["有害", "电池", "药品", "灯管", "油漆", "hazardous", "battery", "medicine"],
    "other": ["其他", "纸巾", "烟头", "口罩", "陶瓷", "other", "tissue", "mask"],
}


def category_catalog():
    """返回给前端的四分类元数据（按固定顺序）。"""
    return [
        {
            "code": code,
            "name": CATEGORY_NAMES[code],
            "en_name": CATEGORY_EN_NAMES[code],
            **CATEGORY_ADVICE[code],
        }
        for code in CATEGORY_ORDER
    ]


def get_advice(code: str) -> dict:
    if code not in CATEGORY_ADVICE:
        raise KeyError(f"未知分类: {code}")
    return {
        "code": code,
        "name": CATEGORY_NAMES[code],
        **CATEGORY_ADVICE[code],
    }
