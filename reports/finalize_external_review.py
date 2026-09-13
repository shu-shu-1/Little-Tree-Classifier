"""Materialize annotations made by visually inspecting review sheets 01–04.

This script contains fixed human-readable visual judgments; no model prediction
or folder-name label is used.  Personal source images are only read, never changed.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
candidates = json.loads((OUT / "external_review_candidates.json").read_text(encoding="utf-8"))
annotations = {}

def mark(ids, label, reason, accepted=None):
    for i in ids:
        annotations[i] = {"label": label, "accepted_labels": accepted if accepted is not None else ([label] if label != "unknown" else []), "review_reason": reason}

mark([0], "city", "实拍传统宫殿建筑门额与屋檐占主体；不是自然风景。")
mark([1], "city", "实拍博物馆入口正立面，主体为建筑。")
mark([2], "city", "传统建筑屋檐为明确主体，前景树木作陪衬。")
mark([3], "city", "城墙城门与现代楼宇，属于建筑/城市景观。")
mark([4], "city", "夜间亮灯宝塔为主体，属于建筑景观。")
mark([5], "city", "灯笼及夜间传统牌坊为主体，属于建筑景观。")
mark([6, 7], "unknown", "真人旅行合影/人物照片，现有八类不包含人物摄影；不能因旅行目录标成风景。")
mark([8], "unknown", "室内考古文物展坑与成排陶俑，现有八类无文物/展品类。")
mark([9, 10], "unknown", "大量游客占据前景的随拍，背景喷水及建筑不构成清晰壁纸主题。")
mark([11], "city", "实拍城墙屋檐与树木；建筑和自然元素混合。", ["city", "nature"])
mark([12, 13], "unknown", "欧美奇幻游戏宣传插画；非明确日式动漫，八类缺少通用游戏/奇幻插画类别。")
mark([14], "unknown", "网页小游戏标题宣传图，含大面积文字和图标，八类没有应用/游戏海报类。")
mark([15, 16], "unknown", "方块体素游戏角色/立体场景宣传图，八类没有通用游戏/像素艺术类。")
mark(range(17, 31), "unknown", "室内真人自拍/合影，现有八类不包含人物摄影。")
mark([31], "nature", "野生老虎与草丛，属于自然动物；未将野生动物当作家养宠物。")
mark([32], "city", "密集历史城镇建筑与塔楼为画面中心，外围为乡村山地。", ["city", "landscape"])
mark([33], "landscape", "峡谷石柱地貌与晚霞的广角自然风光。")
mark([34], "landscape", "沙洲、浅海与大海组成的海岸航拍风景。")
mark([35], "nature", "池塘荷叶间青蛙特写，属于自然动植物。")
mark([36], "landscape", "山地湖泊、树木和倒影构成的广角风景。", ["landscape", "nature"])
mark([37], "city", "古代建筑立柱和装饰天花板的内部建筑摄影。")
mark([38], "landscape", "旷野丘陵中城堡遗迹的环境远景，兼具建筑主题。", ["landscape", "city"])
mark([39], "landscape", "冰湖、山脉与晚霞组成的广角风景。")
mark([40], "city", "石窟造像和人工雕刻建筑遗址为主体。")
mark([41], "anime", "粉发动漫角色插画，具明显动漫线条和面部风格。")
mark([42], "unknown", "体素游戏森林场景；动漫、自然、风景三者边界不清，独立留作范围外/歧义样本。")
mark([43], "nature", "花朵、枝叶的浅景深近摄。")
mark([44], "landscape", "富士山、樱花与宝塔构成经典环境风景，建筑也占明显面积。", ["landscape", "city"])
mark([45], "nature", "草原中的两只野生羚羊，属于自然动物；不是家养宠物。")
mark([46, 47, 50, 52], "anime", "具明确日式动漫/游戏二次元面部、服装与线条的人物插画。")
mark([48, 49], "unknown", "软件界面截图，现有八类无界面/文档类。")
mark([51], "unknown", "文字排版海报，现有八类无文字设计类别。")
mark([53], "unknown", "教材/试卷页面摄影，非现有八类图片主题。")
mark([54], "anime", "东亚幻想游戏男性角色插画，非真人摄影；按二次元角色插画口径归类。")
mark([55], "city", "传统中式建筑园林环境画面，建筑为主要可识别主题。", ["city", "landscape"])
mark([56], "unknown", "低分辨率表情图标，非壁纸图片，八类没有表情/图标类。")
mark([57], "anime", "多角色卡通漫画插画海报。")
mark([58], "unknown", "近真人质感女性肖像/渲染，无法仅凭缩略图确定二次元风格；八类没有人物摄影类。")
mark([59], "anime", "粉色长发动漫角色及幻想环境插画。")
mark([60], "nature", "以孤立开花树木为主的绘画，树木占主体，人形很小；按内容属于自然植物。")
mark([61], "unknown", "白底品牌标志，现有八类没有标志/图标类。")
mark([62, 63, 64, 65, 66, 67], "anime", "动漫角色或明显动漫人物剪影构成的插画。")
mark(range(68, 82), "anime", "实际查看后确认：二次元角色/动漫游戏人物为主体，即使背景含花草或风景也保留动漫类别。")
mark([82], "anime", "具有动漫线条、夸张表情和装饰的卡通猫角色插画；不是真实宠物摄影。")
mark(range(83, 96), "anime", "实际查看后确认：动漫/东亚幻想游戏角色插画或Q版角色，非真实自然/人物摄影。")

assert set(annotations) == set(range(len(candidates)))
rows = []
hash_to_ids = defaultdict(list)
for item in candidates:
    row = dict(item)
    row.update(annotations[item["id"]])
    row["split"] = "external_test"
    row["review_method"] = "manual_visual_contact_sheet"
    row["review_evidence"] = str(OUT / f"external_review_sheet_{item['id']//25+1:02d}.jpg")
    row["review_evidence_cell"] = item["id"] % 25
    row["sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
    row["eligible_for_closed_set"] = row["label"] != "unknown"
    row["do_not_train"] = True
    if row["id"] in {31, 35, 45}:
        row["domain_gap_note"] = "野生动物按内容归nature；现有nature训练图偏昆虫/花卉，此处含大哺乳动物/两栖动物分布差异。"
    hash_to_ids[row["sha256"]].append(row["id"])
    rows.append(row)

known_hashes = set(hash_to_ids)
train_overlap = defaultdict(list)
for folder in [ROOT / "data" / "raw", ROOT / "data" / "split"]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in known_hashes:
                train_overlap[digest].append(str(path))
for row in rows:
    row["exact_duplicate_ids"] = [i for i in hash_to_ids[row["sha256"]] if i != row["id"]]
    row["exact_training_overlap"] = train_overlap.get(row["sha256"], [])
    if row["exact_training_overlap"]:
        row["eligible_for_closed_set"] = False

(OUT / "external_reviewed.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
near_report_path = OUT / "external_nearduplicate_candidates.json"
near_report = json.loads(near_report_path.read_text(encoding="utf-8")) if near_report_path.exists() else None
summary = {
    "reviewed_images": len(rows),
    "label_counts": dict(Counter(row["label"] for row in rows)),
    "known_class_images": sum(row["eligible_for_closed_set"] for row in rows),
    "eligible_label_counts": dict(Counter(row["label"] for row in rows if row["eligible_for_closed_set"])),
    "sampling_groups": dict(Counter(row["sampling_group"] for row in rows)),
    "exact_duplicate_groups": [ids for ids in hash_to_ids.values() if len(ids)>1],
    "exact_training_overlap_count": sum(bool(row["exact_training_overlap"]) for row in rows),
    "perceptual_duplicate_check": {
        "report": str(OUT / "external_nearduplicate_candidates.json"),
        "method": "64-bit DCT pHash; Hamming <= 6 candidate review threshold",
        "training_images_scanned": near_report["training_images_scanned"] if near_report else None,
        "candidate_pairs": len(near_report["matches"]) if near_report else None,
        "scope": "SHA256去重后的eligible external images；详见感知哈希检查脚本与报告",
        "limitation": "阈值内无候选不保证无强裁剪、镜像或较大编辑的重复。",
    },
    "limitations": [
        "仅作固定外部测试，不得用于训练、调参、自动清洗或挑选预测阈值。",
        "八类中只覆盖 anime/city/landscape/nature；pets/cars/space/abstract 未覆盖，不能宣称八类泛化准确率。",
        "人物摄影、截图、文字及无明确对应类的游戏宣传图标为 unknown，不纳入闭集准确率；可单列拒识表现。",
        "primary label用于严格准确率，accepted_labels用于另报包含语义重叠的宽松准确率，不得混为一个指标。",
        "SHA256发现29张原始集重叠并排除；剩余35张与4315张raw/split的pHash Hamming<=6无候选。该检查仍可能漏掉大幅裁剪或编辑。",
        "city 包括建筑/历史建筑，nature 包括野生动物与植物，pets 指家养宠物。",
    ],
}
(OUT / "external_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=True, indent=2))
