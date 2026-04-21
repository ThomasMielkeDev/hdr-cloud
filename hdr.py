import cv2
import numpy as np


def fuse(images):

    import cv2
    import numpy as np

    # --- STEP 1: get base size ---
    h, w = images[0].shape[:2]

    cleaned = []

    for img in images:

        if img is None:
            continue

        # --- force same size ---
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        # --- force 3 channels ---
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # --- convert to float32 (CRITICAL) ---
        img = img.astype(np.float32) / 255.0

        cleaned.append(img)

    if len(cleaned) < 2:
        raise ValueError("Not enough valid images after cleanup")

    merge = cv2.createMergeMertens()

    return merge.process(cleaned)

def enhance(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def grade(img, vividness, sky_boost):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # global vibrance
    s = np.clip(s * vividness, 0, 255)

    # sky boost (blue range)
    sky_mask = (h > 90) & (h < 140)
    s[sky_mask] = np.clip(s[sky_mask] * sky_boost, 0, 255)

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process(images, vividness=1.2, sky_boost=1.3):

    images = [img for img in images if img is not None]

    if len(images) < 2:
        raise ValueError("Need at least 2 images")

    fused = fuse(images)

    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    fused = enhance(fused)

    return grade(fused, vividness, sky_boost)
