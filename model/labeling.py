"""标签归类规则：把数据集子类别 / 文件夹名映射到四分类。

适用于公开数据集常见的英文目录（paper、battery 等）与中文目录
（纸箱、果皮、过期药品等）。对于纯数字目录（如华为云垃圾数据集），
请在 data/label_mapping.json 中显式配置，或提供 garbage_label.txt。
"""

from constants import CATEGORY_NAMES, CATEGORY_ORDER

CATEGORY_ALIASES = {
    "recyclable": ["recyclable", "recycle", "可回收物", "可回收", "可回收垃圾", "蓝桶"],
    "kitchen": ["kitchen", "food", "food waste", "厨余垃圾", "厨余", "湿垃圾", "绿桶"],
    "hazardous": ["hazardous", "harmful", "有害垃圾", "有害", "红桶"],
    "other": ["other", "residual", "residual waste", "其他垃圾", "其他", "干垃圾", "灰桶"],
}

# 关键词按长度从长到短匹配，避免“纸巾”被“纸”误判为可回收物
_KEYWORD_RULES = [
    # 其他垃圾（优先匹配易混淆项）
    ("一次性快餐盒", "other"), ("一次性餐具", "other"), ("一次性", "other"),
    ("污损塑料", "other"), ("污染纸", "other"), ("纸巾", "other"), ("卫生纸", "other"),
    ("纸尿裤", "other"), ("尿布", "other"), ("湿巾", "other"), ("保鲜膜", "other"),
    ("口罩", "other"), ("烟头", "other"), ("烟蒂", "other"), ("牙签", "other"),
    ("竹筷", "other"), ("破碎花盆", "other"), ("陶瓷", "other"), ("贝壳", "other"),
    ("大骨头", "other"), ("尘土", "other"), ("灰土", "other"), ("创可贴", "other"),
    ("tissue", "other"), ("diaper", "other"), ("mask", "other"), ("cigarette", "other"),
    ("ceramic", "other"), ("straw", "other"),
    # 厨余垃圾
    ("剩饭剩菜", "kitchen"), ("剩饭", "kitchen"), ("剩菜", "kitchen"),
    ("水果果皮", "kitchen"), ("水果果肉", "kitchen"), ("西瓜皮", "kitchen"),
    ("香蕉皮", "kitchen"), ("苹果核", "kitchen"), ("果皮", "kitchen"), ("果核", "kitchen"),
    ("菜叶", "kitchen"), ("菜根", "kitchen"), ("蛋壳", "kitchen"), ("茶叶渣", "kitchen"),
    ("咖啡渣", "kitchen"), ("鱼骨", "kitchen"), ("虾壳", "kitchen"), ("骨头", "kitchen"),
    ("食物", "kitchen"), ("厨余", "kitchen"), ("蔬菜", "kitchen"), ("调料", "kitchen"),
    ("food", "kitchen"), ("biological", "kitchen"), ("fruit", "kitchen"),
    ("vegetable", "kitchen"), ("banana", "kitchen"), ("apple", "kitchen"),
    ("orange", "kitchen"), ("eggshell", "kitchen"), ("leftover", "kitchen"),
    # 有害垃圾
    ("过期药品", "hazardous"), ("充电电池", "hazardous"), ("蓄电池", "hazardous"),
    ("水银温度计", "hazardous"), ("荧光灯", "hazardous"), ("杀虫剂", "hazardous"),
    ("电池", "hazardous"), ("充电宝", "hazardous"), ("药品", "hazardous"),
    ("药片", "hazardous"), ("药瓶", "hazardous"), ("灯管", "hazardous"),
    ("灯泡", "hazardous"), ("油漆", "hazardous"), ("农药", "hazardous"),
    ("水银", "hazardous"), ("温度计", "hazardous"), ("消毒剂", "hazardous"),
    ("化学品", "hazardous"), ("battery", "hazardous"), ("medicine", "hazardous"),
    ("drug", "hazardous"), ("lamp", "hazardous"), ("bulb", "hazardous"),
    ("paint", "hazardous"), ("pesticide", "hazardous"), ("mercury", "hazardous"),
    ("chemical", "hazardous"),
    # 可回收物
    ("矿泉水瓶", "recyclable"), ("饮料瓶", "recyclable"), ("洗发水瓶", "recyclable"),
    ("调料瓶", "recyclable"), ("食用油桶", "recyclable"), ("塑料瓶", "recyclable"),
    ("塑料玩具", "recyclable"), ("塑料衣架", "recyclable"), ("塑料", "recyclable"),
    ("纸盒纸箱", "recyclable"), ("快递纸袋", "recyclable"), ("纸箱", "recyclable"),
    ("纸板", "recyclable"), ("报纸", "recyclable"), ("书本", "recyclable"),
    ("杂志", "recyclable"), ("纸张", "recyclable"), ("纸袋", "recyclable"),
    ("易拉罐", "recyclable"), ("饮料罐", "recyclable"), ("金属食品罐", "recyclable"),
    ("金属日用品", "recyclable"), ("金属", "recyclable"), ("铝罐", "recyclable"),
    ("铁罐", "recyclable"), ("罐头盒", "recyclable"), ("玻璃瓶", "recyclable"),
    ("玻璃", "recyclable"), ("牛奶盒", "recyclable"), ("旧衣服", "recyclable"),
    ("毛绒玩具", "recyclable"), ("织物", "recyclable"), ("布料", "recyclable"),
    ("旧书", "recyclable"), ("鞋子", "recyclable"), ("旧手机", "recyclable"),
    ("电子产品", "recyclable"), ("泡沫塑料", "recyclable"), ("插头电线", "recyclable"),
    ("cardboard", "recyclable"), ("newspaper", "recyclable"), ("paper", "recyclable"),
    ("plastic", "recyclable"), ("metal", "recyclable"), ("glass", "recyclable"),
    ("bottle", "recyclable"), ("can", "recyclable"), ("clothes", "recyclable"),
    ("shoe", "recyclable"), ("fabric", "recyclable"), ("textile", "recyclable"),
]

_KEYWORD_RULES.sort(key=lambda item: len(item[0]), reverse=True)


def _normalize(text: str) -> str:
    return str(text).strip().lower().replace("_", " ").replace("-", " ")


_ALIAS_TO_CODE = {
    _normalize(alias): code
    for code, aliases in CATEGORY_ALIASES.items()
    for alias in aliases
}


def to_category_code(value: str) -> str | None:
    """把中文/英文分类名统一成内部编码。"""
    normalized = _normalize(value)
    if normalized in _ALIAS_TO_CODE:
        return _ALIAS_TO_CODE[normalized]
    if normalized in CATEGORY_ORDER:
        return normalized
    return None


def resolve_category(label: str, explicit_map: dict | None = None) -> str | None:
    """将子类别标签解析为四分类编码。

    优先级：显式映射 > 分类别名 > 关键词（长词优先）。
    """
    if not label:
        return None

    raw = str(label).strip()
    if explicit_map:
        for key in (raw, raw.lower(), raw.upper()):
            if key in explicit_map:
                code = to_category_code(explicit_map[key])
                if code:
                    return code

    code = to_category_code(raw)
    if code:
        return code

    normalized = _normalize(raw)
    for keyword, category in _KEYWORD_RULES:
        if keyword in normalized:
            return category
    return None


def category_name(code: str) -> str:
    return CATEGORY_NAMES.get(code, code)
