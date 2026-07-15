"""
Class mapping: model class names → backend categoryCode values.

Both YOLO11 and RF-DETR models may produce different raw class names.
This module normalizes them to the standard PMS defect category codes.
"""

from typing import Optional, Dict

# ── Mapping table ──
# Keys are lowercased model output class names.
# Values are the backend DefectCategory.Code used by the ASP.NET Core API.
#
# Standard category codes:
#   CI  = Cracked Insulator  (Emergency: True)
#   DS  = Damaged Splice     (Emergency: False)
#   VE  = Vegetation Encroachment (Emergency: False)
#   CC  = Corroded Conductor
#   BW  = Bird/Wildlife Damage
#   SA  = Sagging Line
#   ML  = Missing Label
#   OT  = Other

CLASS_TO_CATEGORY: Dict[str, str] = {
    # ── Insulator defects ──
    "broken-disc": "CI",
    "broken-glass": "CI",
    "broken_disc": "CI",
    "broken_glass": "CI",
    "cracked-insulator": "CI",
    "cracked_insulator": "CI",
    "damaged": "CI",
    "damage": "CI",
    "crack": "CI",
    "flashover": "CI",
    "pollution-flashover": "CI",
    "pollution_flashover": "CI",
    "dirt-insulator": "CI",
    "dirt_insulator": "CI",
    "dirt": "CI",
    "pollution": "CI",
    # ── Splice / Connection defects ──
    "disconnected": "DS",
    "misroute": "DS",
    "damaged-splice": "DS",
    "damaged_splice": "DS",
    "splice": "DS",
    # ── Vegetation ──
    "vegetation": "VE",
    "vegetation-encroachment": "VE",
    "vegetation_encroachment": "VE",
    # ── Corrosion ──
    "corrosion": "CC",
    "corroded": "CC",
    "corroded-conductor": "CC",
    "corroded_conductor": "CC",
    "rust": "CC",
    # ── Bird / Wildlife ──
    "bird-nest": "BW",
    "bird_nest": "BW",
    "bird": "BW",
    "wildlife": "BW",
    # ── Sagging ──
    "sagging": "SA",
    "sag": "SA",
    # ── Other defects ──
    "missing-label": "ML",
    "missing_label": "ML",
    "other": "OT",
}

# Labels that indicate a normal / non-defective state → should be skipped
NORMAL_LABELS = frozenset([
    "normal",
    "clean",
    "clean-insulator",
    "clean_insulator",
    "good",
    "healthy",
    "ok",
    "insulator",  # raw YOLO class for a healthy insulator
])


def map_class_to_category(label: str) -> Optional[str]:
    """Map a model class label to a backend defect category code.

    Args:
        label: Raw class name from model output.

    Returns:
        Category code string (e.g. 'CI', 'DS', 'VE'), or None if the
        label represents a normal/non-defective state.
    """
    norm = label.lower().strip().replace(" ", "-")

    # Skip normal labels
    if norm in NORMAL_LABELS:
        return None

    # Direct lookup
    if norm in CLASS_TO_CATEGORY:
        return CLASS_TO_CATEGORY[norm]

    # Fuzzy fallback — check if any key is a substring
    for key, code in CLASS_TO_CATEGORY.items():
        if key in norm or norm in key:
            return code

    # If we can't match but it's not a known normal label,
    # assume it's a defect and map to "Other"
    return "OT"
