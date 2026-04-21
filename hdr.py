import cv2
import numpy as np


# ----------------------------
# SAFE IMAGE NORMALIZATION
# ----------------------------
def normalize_images(images):

    cleaned = []

    # base size
    base = images[0]
    h, w = base.shape[:2]

    for img in images:

        if img is None:
            continue

        # resize to match base
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # force 3-channel BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ensure uint8 BEFORE conversion
        img = np.clip(img, 0, 255).astype(np.uint8)

        cleaned.append(img)

    if len(cleaned) < 2:
        raise ValueError("Not enough valid images after normalization")

    return cleaned


# ----------------------------
# HDR FUSION (SAFE VERSION)
# ----------------------------
def fuse(images):

    import cv2
    import numpy as np

    cleaned = []

    # 1. validate + normalize inputs
    for img in images:
        if img is None:
            continue

        # force BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        cleaned.append(img)

    if len(cleaned) < 2:
        raise ValueError("Not enough valid images")

    # 2. force SAME SIZE (critical)
    h, w = cleaned[0].shape[:2]

    normalized = []
    for img in cleaned:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        normalized.append(img)

    # 3. exposure fusion (NUMPY, not OpenCV C++ merge)
    result = np.zeros_like(normalized[0], dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)

    for img in normalized:

        # luminance-based weight
        lum = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        weight = np.exp(-((lum - 0.5) ** 2) / 0.08)

        weight = weight[..., None]

        result += img * weight
        weight_sum += weight

    result /= np.clip(weight_sum, 1e-6, None)

    return np.clip(result / 255.0, 0, 1)

# ----------------------------
# ENHANCEMENT (LIGHTROOM STYLE)
# ----------------------------
def enhance(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------
# COLOR GRADING (SLIDERS)
# ----------------------------
def grade(img, vividness, sky_boost):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = np.clip(s * vividness, 0, 255)

    sky_mask = (h > 90) & (h < 140)
    s[sky_mask] = np.clip(s[sky_mask] * sky_boost, 0, 255)

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def process(images, vividness=1.2, sky_boost=1.3):

    if images is None or len(images) == 0:
        raise ValueError("No images provided")

    images = [img for img in images if img is not None]

    if len(images) < 2:
        raise ValueError("Need at least 2 images")

    fused = fuse(images)

    if fused is None:
        raise ValueError("HDR fusion failed")

    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    fused = enhance(fused)

    return grade(fused, vividness, sky_boost)
