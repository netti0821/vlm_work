#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_pandaset_to_dtr_jsonl.py

PandaSet の camera画像 + annotations/cuboids/*.pkl から、
distill_then_replace(DTR) 環境用の train.jsonl / eval.jsonl を生成します。

今回版の追加点:
- class別 min_area に対応
  例:
    --min_area_by_class "car=500,person=200,bicycle=200,motorcycle=200,traffic_sign=30,traffic_light=30"
- 小さい遠方車両を落としつつ、traffic_sign / traffic_light は小さくても残しやすくする。
- max_objects_per_image で切る前に大きいbboxを優先するオプションを追加
  例:
    --prefer_large_bbox_when_limit

実行例:
python tools/convert_pandaset_to_dtr_jsonl.py ^
  --pandaset_root .\\datasets\\pandaset ^
  --out_root .\\data\\automotive_vlm_pandaset ^
  --cameras front_camera ^
  --frame_stride 5 ^
  --eval_ratio 0.1 ^
  --min_area_by_class "car=500,person=200,bicycle=200,motorcycle=200,traffic_sign=30,traffic_light=30" ^
  --max_objects_per_image 20 ^
  --prefer_large_bbox_when_limit
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pandaset import DataSet
from pandaset.geometry import center_box_to_corners, projection


LABEL_MAP = {
    "Car": "car",
    "Pickup Truck": "car",
    "Medium-sized Truck": "car",
    "Semi-truck": "car",
    "Other Vehicle - Uncommon": "car",
    "Other Vehicle - Construction Vehicle": "car",
    "Towed Object": "car",
    "Bus": "car",

    "Pedestrian": "person",
    "Pedestrian with Object": "person",

    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Motorized Scooter": "motorcycle",
    "Personal Mobility Device": "motorcycle",

    "Signs": "traffic_sign",
    "Traffic Sign": "traffic_sign",
    "Traffic Signs": "traffic_sign",
    "Construction Signs": "traffic_sign",

    "Traffic Light": "traffic_light",
    "Traffic Lights": "traffic_light",

    # "Cones" は現行DTR対象classに無いため除外
}

VALID_DTR_CLASSES = {
    "car",
    "traffic_light",
    "traffic_sign",
    "person",
    "bicycle",
    "motorcycle",
}

DEFAULT_INSTRUCTION = (
    "信号機、交通標識、車、人、自転車、バイクを分類し、CSV形式で出力してください。"
)

DEFAULT_MIN_AREA_BY_CLASS = {
    # normalized_1000座標上の面積。
    # 例: bbox=[100,100,150,150] の面積は 50*50=2500。
    #
    # carは遠方の小さい車を落としやすくするため大きめ。
    # traffic_sign / traffic_light は遠方でも残したいので小さめ。
    "car": 500.0,
    "person": 200.0,
    "bicycle": 200.0,
    "motorcycle": 200.0,
    "traffic_sign": 30.0,
    "traffic_light": 30.0,
}


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_min_area_by_class(text: str | None) -> dict[str, float]:
    """
    "car=500,person=200,traffic_sign=30" を dict に変換する。
    空文字なら DEFAULT_MIN_AREA_BY_CLASS を返す。
    """
    if text is None or str(text).strip() == "":
        return dict(DEFAULT_MIN_AREA_BY_CLASS)

    result = dict(DEFAULT_MIN_AREA_BY_CLASS)

    for item in parse_csv_list(text):
        if "=" not in item:
            raise ValueError(
                f"Invalid --min_area_by_class item: {item}. Expected class=value"
            )
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in VALID_DTR_CLASSES:
            raise ValueError(
                f"Invalid class in --min_area_by_class: {k}. "
                f"valid={sorted(VALID_DTR_CLASSES)}"
            )

        result[k] = float(v)

    return result


def to_jsonl_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm1000_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[int]:
    return [
        int(round(x1 / max(1, w) * 1000)),
        int(round(y1 / max(1, h) * 1000)),
        int(round(x2 / max(1, w) * 1000)),
        int(round(y2 / max(1, h) * 1000)),
    ]


def clip_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(w - 1), float(x1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    y2 = max(0.0, min(float(h - 1), float(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_area_xyxy(bbox: list[int] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = bbox_area_xyxy(a) + bbox_area_xyxy(b) - inter
    return 0.0 if union <= 0 else inter / union


def nms_objects(objects: list[dict[str, Any]], iou_threshold: float = 0.85) -> list[dict[str, Any]]:
    if iou_threshold <= 0:
        return objects

    sorted_objs = sorted(objects, key=lambda o: bbox_area_xyxy(o["bbox"]), reverse=True)
    kept: list[dict[str, Any]] = []

    for obj in sorted_objs:
        duplicate = False
        for kept_obj in kept:
            if obj["class"] != kept_obj["class"]:
                continue
            if bbox_iou(obj["bbox"], kept_obj["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)

    kept.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["class"]))
    return kept


def get_cuboid_files(pandaset_root: str | Path, seq_id: str) -> list[Path]:
    cuboid_dir = Path(pandaset_root) / str(seq_id) / "annotations" / "cuboids"
    files = list(cuboid_dir.glob("*.pkl"))

    def sort_key(p: Path) -> int | str:
        try:
            return int(p.stem)
        except Exception:
            return p.stem

    return sorted(files, key=sort_key)


def read_cuboids_frame(cuboid_files: list[Path], frame_idx: int) -> pd.DataFrame:
    if frame_idx < 0 or frame_idx >= len(cuboid_files):
        raise IndexError(f"cuboid_files length={len(cuboid_files)}, frame_idx={frame_idx}")

    df = pd.read_pickle(cuboid_files[frame_idx])
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"cuboid pkl is not DataFrame: {cuboid_files[frame_idx]}")

    return df


def get_camera_num_frames(cam: Any) -> int:
    lengths = []
    for attr in ["data", "poses", "timestamps"]:
        v = getattr(cam, attr, None)
        if v is not None:
            try:
                lengths.append(len(v))
            except Exception:
                pass

    if not lengths:
        raise RuntimeError("Cannot determine camera frame length. Did you call cam.load()?")

    return min(lengths)


def safe_camera_load(cam: Any, seq_id: str, cam_name: str) -> bool:
    try:
        cam.load()
        return True
    except Exception as e:
        print(f"[WARN] camera load failed: seq={seq_id}, camera={cam_name}, error={repr(e)}")
        return False


def row_to_box(row: pd.Series) -> list[float]:
    candidates = [
        ("position.x", "position.y", "position.z", "dimensions.x", "dimensions.y", "dimensions.z", "yaw"),
        ("x", "y", "z", "dx", "dy", "dz", "yaw"),
        ("center.x", "center.y", "center.z", "size.x", "size.y", "size.z", "yaw"),
    ]

    for cols in candidates:
        if all(c in row.index for c in cols):
            vals = [float(row[c]) for c in cols]
            if any(math.isnan(v) for v in vals):
                raise ValueError("cuboid row contains NaN")
            return vals

    raise KeyError(f"Unsupported cuboid columns: {list(row.index)}")


def should_keep_by_camera_used(row: pd.Series, cam_name: str, enable_filter: bool) -> bool:
    if not enable_filter:
        return True

    if "camera_used" not in row.index:
        return True

    val = row.get("camera_used")
    if pd.isna(val):
        return True

    return cam_name in str(val)


def convert_label(raw_label: Any) -> str | None:
    if raw_label is None or pd.isna(raw_label):
        return None

    mapped = LABEL_MAP.get(str(raw_label))
    if mapped not in VALID_DTR_CLASSES:
        return None

    return mapped


def project_cuboid_to_bbox(
    row: pd.Series,
    image: Image.Image,
    camera_pose: Any,
    camera_intrinsics: Any,
    min_projected_points: int,
    filter_outliers: bool,
) -> tuple[float, float, float, float] | None:
    image_w, image_h = image.size

    box = row_to_box(row)
    corners_3d = center_box_to_corners(box)

    try:
        proj_result = projection(
            lidar_points=corners_3d,
            camera_data=image,
            camera_pose=camera_pose,
            camera_intrinsics=camera_intrinsics,
            filter_outliers=filter_outliers,
        )
    except TypeError:
        proj_result = projection(
            corners_3d,
            image,
            camera_pose,
            camera_intrinsics,
            filter_outliers=filter_outliers,
        )

    if isinstance(proj_result, tuple):
        points_2d = proj_result[0]
    else:
        points_2d = proj_result

    if points_2d is None:
        return None

    points_2d = np.asarray(points_2d)

    if points_2d.ndim != 2:
        return None

    if points_2d.shape[1] != 2 and points_2d.shape[0] == 2:
        points_2d = points_2d.T

    if points_2d.shape[1] < 2 or points_2d.shape[0] < min_projected_points:
        return None

    xs = points_2d[:, 0]
    ys = points_2d[:, 1]

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))

    return clip_bbox(x1, y1, x2, y2, image_w, image_h)


def cuboids_to_objects(
    seq: Any,
    cam_name: str,
    frame_idx: int,
    image: Image.Image,
    cuboid_files: list[Path],
    min_area_norm1000: float,
    min_area_by_class: dict[str, float],
    min_projected_points: int,
    use_camera_used_filter: bool,
    projection_filter_outliers: bool,
    label_counter: Counter,
    skip_counter: Counter,
) -> list[dict[str, Any]]:
    cam = seq.camera[cam_name]
    image_w, image_h = image.size

    try:
        camera_pose = cam.poses[frame_idx]
        camera_intrinsics = cam.intrinsics
    except Exception as e:
        print(f"[WARN] camera metadata failed: camera={cam_name}, frame={frame_idx}, error={repr(e)}")
        return []

    try:
        df = read_cuboids_frame(cuboid_files, frame_idx)
    except Exception as e:
        print(f"[WARN] cuboid frame load failed: frame={frame_idx}, error={repr(e)}")
        return []

    objects: list[dict[str, Any]] = []
    seen_uuid: set[str] = set()

    for _, row in df.iterrows():
        raw_label = row.get("label")
        label_counter[str(raw_label)] += 1

        cls = convert_label(raw_label)
        if cls is None:
            skip_counter["skip_label"] += 1
            continue

        if not should_keep_by_camera_used(row, cam_name, use_camera_used_filter):
            skip_counter["skip_camera_used"] += 1
            continue

        # projection成功前には seen_uuid に入れない。
        # 失敗したcuboidが同uuidの後続候補を潰すのを避ける。
        uuid_s = None
        uuid = row.get("uuid", None)
        if uuid is not None and not pd.isna(uuid):
            uuid_s = str(uuid)
            if uuid_s in seen_uuid:
                skip_counter["skip_duplicate_uuid"] += 1
                continue

        try:
            clipped = project_cuboid_to_bbox(
                row=row,
                image=image,
                camera_pose=camera_pose,
                camera_intrinsics=camera_intrinsics,
                min_projected_points=min_projected_points,
                filter_outliers=projection_filter_outliers,
            )
        except Exception:
            skip_counter["skip_projection_error"] += 1
            continue

        if clipped is None:
            skip_counter["skip_no_projection"] += 1
            continue

        x1, y1, x2, y2 = clipped
        bbox = norm1000_bbox(x1, y1, x2, y2, image_w, image_h)
        area = bbox_area_xyxy(bbox)

        # class別min_area。指定が無いclassは global min_area_norm1000 を使う。
        min_area = float(min_area_by_class.get(cls, min_area_norm1000))

        if area < min_area:
            skip_counter[f"skip_small_area_{cls}"] += 1
            continue

        if uuid_s is not None:
            seen_uuid.add(uuid_s)

        objects.append(
            {
                "class": cls,
                "bbox": bbox,
                # 後段のsort用。JSONL出力前に削除する。
                "_area": area,
            }
        )

    return objects


def cleanup_objects_for_json(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for obj in objects:
        cleaned.append(
            {
                "class": obj["class"],
                "bbox": obj["bbox"],
            }
        )
    return cleaned


def split_records(
    records: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
    split_by_sequence: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    rng = random.Random(seed)

    if split_by_sequence:
        seq_ids = sorted({str(r.get("meta", {}).get("sequence", "")) for r in records})
        rng.shuffle(seq_ids)

        n_eval_seq = max(1, int(round(len(seq_ids) * eval_ratio))) if len(seq_ids) > 1 else 1
        eval_seq = set(seq_ids[:n_eval_seq])

        train_records = [r for r in records if str(r.get("meta", {}).get("sequence", "")) not in eval_seq]
        eval_records = [r for r in records if str(r.get("meta", {}).get("sequence", "")) in eval_seq]

        if train_records and eval_records:
            return train_records, eval_records

    idxs = list(range(len(records)))
    rng.shuffle(idxs)

    if len(records) == 1:
        return records, records

    n_eval = max(1, int(round(len(records) * eval_ratio)))
    n_eval = min(n_eval, len(records) - 1)

    eval_idx = set(idxs[:n_eval])
    train_records = [r for i, r in enumerate(records) if i not in eval_idx]
    eval_records = [r for i, r in enumerate(records) if i in eval_idx]

    return train_records, eval_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pandaset_root", required=True)
    parser.add_argument("--out_root", default="./data/automotive_vlm_pandaset")
    parser.add_argument("--cameras", default="front_camera")
    parser.add_argument("--sequences", default="", help="comma separated sequence ids. empty means all")
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    parser.add_argument(
        "--min_area_norm1000",
        type=float,
        default=20.0,
        help="fallback normalized_1000 bbox area threshold",
    )
    parser.add_argument(
        "--min_area_by_class",
        default="car=500,person=200,bicycle=200,motorcycle=200,traffic_sign=30,traffic_light=30",
        help=(
            "class specific min area. "
            "Example: car=500,person=200,bicycle=200,motorcycle=200,traffic_sign=30,traffic_light=30"
        ),
    )
    parser.add_argument("--min_projected_points", type=int, default=3)
    parser.add_argument("--max_objects_per_image", type=int, default=30)
    parser.add_argument("--nms_iou", type=float, default=0.85)
    parser.add_argument("--use_camera_used_filter", action="store_true")
    parser.add_argument("--split_by_sequence", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--projection_filter_outliers",
        action="store_true",
        help=(
            "Use pandaset projection(filter_outliers=True). "
            "Default is False to keep edge/near partially visible objects."
        ),
    )
    parser.add_argument(
        "--prefer_large_bbox_when_limit",
        action="store_true",
        help=(
            "When max_objects_per_image is applied, keep larger bbox first. "
            "Useful to drop far small objects."
        ),
    )

    args = parser.parse_args()

    pandaset_root = Path(args.pandaset_root)
    out_root = Path(args.out_root)
    image_out = out_root / "images"

    if not pandaset_root.exists():
        raise FileNotFoundError(f"pandaset_root not found: {pandaset_root}")

    if not args.dry_run:
        ensure_dir(image_out)

    cameras = parse_csv_list(args.cameras)
    min_area_by_class = parse_min_area_by_class(args.min_area_by_class)

    dataset = DataSet(str(pandaset_root))

    if args.sequences:
        seq_ids = parse_csv_list(args.sequences)
    else:
        try:
            seq_ids = list(dataset.sequences())
        except TypeError:
            seq_ids = list(dataset.sequences)

    print(f"[INFO] pandaset_root={pandaset_root}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] sequences={len(seq_ids)}")
    print(f"[INFO] cameras={cameras}")
    print(f"[INFO] min_area_by_class={min_area_by_class}")
    print(f"[INFO] projection_filter_outliers={args.projection_filter_outliers}")

    records: list[dict[str, Any]] = []
    label_counter: Counter = Counter()
    class_counter: Counter = Counter()
    skip_counter: Counter = Counter()
    per_seq_record_count: dict[str, int] = defaultdict(int)

    for seq_id in tqdm(seq_ids, desc="sequences"):
        seq_id = str(seq_id)
        cuboid_files = get_cuboid_files(pandaset_root, seq_id)

        if not cuboid_files:
            print(f"[WARN] no cuboid pkl files: seq={seq_id}")
            continue

        print(f"[INFO] seq={seq_id}, cuboid_frames={len(cuboid_files)}")

        seq = dataset[seq_id]

        try:
            seq.load()
        except Exception:
            pass

        for cam_name in cameras:
            if cam_name not in seq.camera:
                print(f"[WARN] camera not found: seq={seq_id}, camera={cam_name}")
                continue

            cam = seq.camera[cam_name]

            if not safe_camera_load(cam, seq_id, cam_name):
                continue

            try:
                cam_frames = get_camera_num_frames(cam)
            except Exception as e:
                print(f"[WARN] camera frame count failed: seq={seq_id}, camera={cam_name}, error={repr(e)}")
                continue

            num_frames = min(cam_frames, len(cuboid_files))

            for frame_idx in range(0, num_frames, max(1, args.frame_stride)):
                try:
                    image = cam[frame_idx].convert("RGB")
                except Exception as e:
                    print(
                        f"[WARN] image load failed: seq={seq_id}, camera={cam_name}, "
                        f"frame={frame_idx}, error={repr(e)}"
                    )
                    continue

                objects = cuboids_to_objects(
                    seq=seq,
                    cam_name=cam_name,
                    frame_idx=frame_idx,
                    image=image,
                    cuboid_files=cuboid_files,
                    min_area_norm1000=args.min_area_norm1000,
                    min_area_by_class=min_area_by_class,
                    min_projected_points=args.min_projected_points,
                    use_camera_used_filter=args.use_camera_used_filter,
                    projection_filter_outliers=args.projection_filter_outliers,
                    label_counter=label_counter,
                    skip_counter=skip_counter,
                )

                if args.nms_iou > 0:
                    objects = nms_objects(objects, iou_threshold=args.nms_iou)

                if args.prefer_large_bbox_when_limit:
                    objects.sort(key=lambda o: float(o.get("_area", bbox_area_xyxy(o["bbox"]))), reverse=True)

                if args.max_objects_per_image and args.max_objects_per_image > 0:
                    objects = objects[: args.max_objects_per_image]

                # 最終出力は見やすいように画像上の位置順へ戻す
                objects.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["class"]))

                if not objects:
                    skip_counter["skip_empty_objects_image"] += 1
                    continue

                for obj in objects:
                    class_counter[obj["class"]] += 1

                objects_for_json = cleanup_objects_for_json(objects)

                image_name = f"pandaset_{seq_id}_{cam_name}_{frame_idx:03d}.jpg"
                image_path = image_out / image_name

                if not args.dry_run:
                    image.save(image_path, quality=95)

                rec = {
                    "image_path": image_name,
                    "instruction": args.instruction,
                    "answer_json": {
                        "source_type": "image",
                        "frame_id": frame_idx,
                        "coord_system": "normalized_1000",
                        "objects": objects_for_json,
                    },
                    "meta": {
                        "dataset": "PandaSet",
                        "sequence": seq_id,
                        "camera": cam_name,
                        "frame_idx": frame_idx,
                    },
                }

                records.append(rec)
                per_seq_record_count[seq_id] += 1

    train_records, eval_records = split_records(
        records=records,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        split_by_sequence=args.split_by_sequence,
    )

    stats = {
        "records": len(records),
        "train": len(train_records),
        "eval": len(eval_records),
        "label_counter_raw": dict(label_counter.most_common()),
        "class_counter_dtr": dict(class_counter.most_common()),
        "skip_counter": dict(skip_counter.most_common()),
        "per_seq_record_count": dict(sorted(per_seq_record_count.items())),
        "min_area_by_class": min_area_by_class,
        "args": vars(args),
    }

    if not args.dry_run:
        ensure_dir(out_root)

        with open(out_root / "train.jsonl", "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(to_jsonl_line(r))

        with open(out_root / "eval.jsonl", "w", encoding="utf-8") as f:
            for r in eval_records:
                f.write(to_jsonl_line(r))

        with open(out_root / "label_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print("records:", len(records))
    print("train:", len(train_records))
    print("eval:", len(eval_records))
    print("images:", image_out)
    print("class_counter_dtr:", dict(class_counter.most_common()))
    print("skip_counter:", dict(skip_counter.most_common()))
    print("min_area_by_class:", min_area_by_class)
    print("label_stats:", out_root / "label_stats.json")


if __name__ == "__main__":
    main()
