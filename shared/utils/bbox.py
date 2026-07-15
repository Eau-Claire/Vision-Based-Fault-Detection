"""
Bounding box normalization utilities.

All coordinates are normalized to [0, 1] relative to image dimensions.
"""


def normalize_bbox(
    x1: float, y1: float, x2: float, y2: float,
    img_width: int, img_height: int,
) -> dict:
    """Convert absolute pixel coordinates to normalized [0, 1] bounding box.

    Args:
        x1, y1: Top-left corner (pixels).
        x2, y2: Bottom-right corner (pixels).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Dict with keys: x, y, width, height — all in [0, 1].
    """
    if img_width <= 0 or img_height <= 0:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}

    nx = max(0.0, min(1.0, float(x1) / img_width))
    ny = max(0.0, min(1.0, float(y1) / img_height))
    nw = max(0.0, min(1.0, float(x2 - x1) / img_width))
    nh = max(0.0, min(1.0, float(y2 - y1) / img_height))

    return {"x": nx, "y": ny, "width": nw, "height": nh}


def clamp_bbox(x: float, y: float, w: float, h: float) -> dict:
    """Clamp already-normalized bbox values to [0, 1].

    Args:
        x, y: Top-left corner (normalized).
        w, h: Width and height (normalized).

    Returns:
        Dict with clamped values.
    """
    return {
        "x": max(0.0, min(1.0, x)),
        "y": max(0.0, min(1.0, y)),
        "width": max(0.0, min(1.0, w)),
        "height": max(0.0, min(1.0, h)),
    }


def calculate_iou(box_a: dict, box_b: dict) -> float:
    """Calculate Intersection over Union for two normalized bounding boxes.

    Each box is a dict with keys: x, y, width, height.
    """
    ax1 = box_a["x"]
    ay1 = box_a["y"]
    ax2 = ax1 + box_a["width"]
    ay2 = ay1 + box_a["height"]

    bx1 = box_b["x"]
    by1 = box_b["y"]
    bx2 = bx1 + box_b["width"]
    by2 = by1 + box_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area
