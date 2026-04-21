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

    if len(images) < 2:
        raise ValueError("Need at least 2 images")

    # -----------------------------
    # STEP 1: FORCE BASE IMAGE
    # -----------------------------
    base = images[0]
    h, w = base.shape[:2]

    cleaned = []

    for img in images:

        if img is None:
            continue

        # FORCE SIZE
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # FORCE CHANNEL COUNT
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # FORCE TYPE (IMPORTANT)
        img = np.ascontiguousarray(img, dtype=np.uint8)

        cleaned.append(img)

    # FINAL SAFETY CHECK (THIS IS KEY)
    for i in cleaned:
        if i.shape != cleaned[0].shape:
            raise ValueError(f"Shape mismatch detected: {i.shape} vs {cleaned[0].shape}")

    # -----------------------------
    # STEP 2: SAFE CONVERSION
    # -----------------------------
    float_images = []
    for img in cleaned:
        f = img.astype(np.float32) / 255.0
        float_images.append(f)

    # -----------------------------
    # STEP 3: MERGE
    # -----------------------------
    merge = cv2.createMergeMertens()

    result = merge.process(float_images)

    return result


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
