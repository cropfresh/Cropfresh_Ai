"""
Digital Twin — Image Diff Analysis
===================================
Computes visual similarity between departure and arrival photos.

Strategy (best-effort, ordered by quality):
  1. SSIM (structural similarity) via scikit-image  — best, requires PIL + skimage
  2. Perceptual hash distance via imagehash         — good, requires imagehash + PIL
  3. Rule-based similarity from grade + defect data — always available as fallback

All photo-based methods gracefully degrade when libraries are absent or
when S3 URLs cannot be fetched in the current environment.
"""

# * DIFF ANALYSIS MODULE
# NOTE: All image-loading operations are timeout-guarded to avoid hanging CI pipelines.
# NOTE: Rule-based fallback produces deterministic results based on grade/defect metadata.

from __future__ import annotations

from typing import Optional

from loguru import logger

# * Grade numeric values for delta computation (higher = better)
GRADE_ORDER: dict[str, int] = {"A+": 4, "A": 3, "B": 2, "C": 1}

# * Max possible grade drop span (A+ → C = 3 levels)
MAX_GRADE_SPAN: int = 3

# * Image fetch timeout in seconds for URL-based analysis
IMAGE_FETCH_TIMEOUT: int = 5


# * ═══════════════════════════════════════════════════════════════
# * Grade & Defect Metrics
# * ═══════════════════════════════════════════════════════════════

def compute_grade_delta(grade_departure: str, grade_arrival: str) -> float:
    """
    Compute quality_delta as a value in [-1.0, 0.0].

    0.0  = no change (or improvement)
    -1.0 = maximum degradation (A+ → C)

    Args:
        grade_departure: Grade at farm gate  (e.g. 'A')
        grade_arrival:   Grade at buyer site (e.g. 'B')

    Returns:
        Float in [-1.0, 0.0].
    """
    dep = GRADE_ORDER.get(grade_departure, 2)
    arr = GRADE_ORDER.get(grade_arrival, 2)
    diff = arr - dep
    return max(-1.0, min(0.0, diff / MAX_GRADE_SPAN))


def compute_new_defects(
    departure_defects: list[str],
    arrival_defects: list[str],
) -> list[str]:
    """
    Return defects present at arrival that were absent at departure.

    Args:
        departure_defects: Defect labels detected at farm gate.
        arrival_defects:   Defect labels detected at buyer site.

    Returns:
        List of newly introduced defect labels.
    """
    departure_set = set(departure_defects)
    return [d for d in arrival_defects if d not in departure_set]


def compute_rule_based_similarity(
    grade_departure: str,
    grade_arrival: str,
    departure_defects: list[str],
    arrival_defects: list[str],
) -> float:
    """
    Compute a similarity proxy [0.0, 1.0] from metadata when images are unavailable.

    Starts at 1.0 and subtracts penalties for grade drops and new defects.

    Args:
        grade_departure: Departure grade string.
        grade_arrival:   Arrival grade string.
        departure_defects: Defects at departure.
        arrival_defects:   Defects at arrival.

    Returns:
        Float in [0.0, 1.0].
    """
    dep_val = GRADE_ORDER.get(grade_departure, 2)
    arr_val = GRADE_ORDER.get(grade_arrival, 2)
    grade_drop = max(0, dep_val - arr_val)
    grade_penalty = grade_drop / MAX_GRADE_SPAN

    new_defect_count = len(compute_new_defects(departure_defects, arrival_defects))
    defect_penalty = min(0.30, new_defect_count * 0.07)

    return round(max(0.0, min(1.0, 1.0 - grade_penalty - defect_penalty)), 4)


# * ═══════════════════════════════════════════════════════════════
# * SSIM Analysis (requires scikit-image + Pillow)
# * ═══════════════════════════════════════════════════════════════

def _load_image_from_url(url: str):  # type: ignore[return]
    """
    Download an image from a URL and return a grayscale numpy array (128×128).

    Returns:
        np.ndarray or None if loading fails / URL is not HTTP(S).
    """
    if not url.startswith(("http://", "https://")):
        return None
    try:
        import urllib.request
        from io import BytesIO

        import numpy as np
        from PIL import Image

        with urllib.request.urlopen(url, timeout=IMAGE_FETCH_TIMEOUT) as resp:
            raw = resp.read()
        img = Image.open(BytesIO(raw)).convert("L")
        img = img.resize((128, 128))
        return np.array(img)
    except Exception as exc:
        logger.debug(f"Image load failed for {url}: {exc}")
        return None


def try_ssim_analysis(
    departure_photo_url: Optional[str],
    arrival_photo_url: Optional[str],
) -> Optional[float]:
    """
    Compute SSIM score between two photo URLs.

    Requires scikit-image and Pillow. Returns None when unavailable
    or when photos cannot be fetched.

    Args:
        departure_photo_url: S3/HTTP URL of departure photo.
        arrival_photo_url:   S3/HTTP URL of arrival photo.

    Returns:
        SSIM score [0.0, 1.0] or None.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        logger.debug("scikit-image not installed — SSIM unavailable")
        return None

    dep_img = _load_image_from_url(departure_photo_url or "")
    arr_img = _load_image_from_url(arrival_photo_url or "")
    if dep_img is None or arr_img is None:
        return None

    try:
        score, _ = structural_similarity(dep_img, arr_img, full=True)
        return float(round(score, 4))
    except Exception as exc:
        logger.debug(f"SSIM computation failed: {exc}")
        return None


# * ═══════════════════════════════════════════════════════════════
# * Perceptual Hash Analysis (requires imagehash + Pillow)
# * ═══════════════════════════════════════════════════════════════

def try_perceptual_hash(
    departure_photo_url: Optional[str],
    arrival_photo_url: Optional[str],
) -> Optional[float]:
    """
    Compute perceptual hash similarity between two photo URLs.

    Requires imagehash and Pillow. Returns None when unavailable
    or when photos cannot be fetched.

    Args:
        departure_photo_url: S3/HTTP URL of departure photo.
        arrival_photo_url:   S3/HTTP URL of arrival photo.

    Returns:
        Similarity score [0.0, 1.0] or None.
    """
    try:
        import imagehash
    except ImportError:
        logger.debug("imagehash not installed — perceptual hash unavailable")
        return None

    def _load_pil(url: str):  # type: ignore[return]
        if not url.startswith(("http://", "https://")):
            return None
        try:
            import urllib.request
            from io import BytesIO

            from PIL import Image

            with urllib.request.urlopen(url, timeout=IMAGE_FETCH_TIMEOUT) as resp:
                return Image.open(BytesIO(resp.read()))
        except Exception as exc:
            logger.debug(f"PIL load failed for {url}: {exc}")
            return None

    dep_img = _load_pil(departure_photo_url or "")
    arr_img = _load_pil(arrival_photo_url or "")
    if dep_img is None or arr_img is None:
        return None

    try:
        # NOTE: pHash distance ranges 0 (identical) → 64 (completely different)
        hash_dep = imagehash.phash(dep_img)
        hash_arr = imagehash.phash(arr_img)
        distance = hash_dep - hash_arr
        return float(round(1.0 - distance / 64.0, 4))
    except Exception as exc:
        logger.debug(f"Perceptual hash computation failed: {exc}")
        return None


# * ═══════════════════════════════════════════════════════════════
# * Unified Similarity Dispatcher
# * ═══════════════════════════════════════════════════════════════

def compute_similarity(
    departure_photos: list[str],
    arrival_photos: list[str],
    grade_departure: str,
    grade_arrival: str,
    departure_defects: list[str],
    arrival_defects: list[str],
) -> tuple[float, str]:
    """
    Compute the best available similarity score with method fallback.

    Attempt order:
      1. SSIM on first photo pair          (method='ssim')
      2. Perceptual hash on first pair     (method='perceptual_hash')
      3. Rule-based from grade/defect data (method='rule_based')

    Args:
        departure_photos:  All departure photo URLs (farmer + agent).
        arrival_photos:    All arrival photo URLs.
        grade_departure:   Departure grade string.
        grade_arrival:     Arrival grade string.
        departure_defects: Defects at departure.
        arrival_defects:   Defects at arrival.

    Returns:
        Tuple of (similarity_score [0.0, 1.0], method_name).
    """
    if departure_photos and arrival_photos:
        dep_url = departure_photos[0]
        arr_url = arrival_photos[0]

        # * Try SSIM first
        ssim_score = try_ssim_analysis(dep_url, arr_url)
        if ssim_score is not None:
            logger.debug(f"SSIM similarity: {ssim_score:.4f}")
            return ssim_score, "ssim"

        # * Fall back to perceptual hash
        phash_score = try_perceptual_hash(dep_url, arr_url)
        if phash_score is not None:
            logger.debug(f"Perceptual hash similarity: {phash_score:.4f}")
            return phash_score, "perceptual_hash"

    # * Rule-based fallback — always available
    rule_score = compute_rule_based_similarity(
        grade_departure, grade_arrival, departure_defects, arrival_defects
    )
    logger.debug(f"Rule-based similarity: {rule_score:.4f}")
    return rule_score, "rule_based"
