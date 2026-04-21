import cv2
import numpy as np


def fuse(images):

    import cv2
    import numpy as np

    if len(images) == 0:
        raise ValueError("No images passed to fuse")

    # --- STEP 1: force base size ---
    base = images[0]
    h, w = base.shape[:2]

    cleaned = []

    for idx, img in enumerate(images):

        if img is None:
            continue

        # --- fix size ---
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # --- fix grayscale ---
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # --- fix alpha channel ---
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # --- FORCE TYPE CONSISTENCY ---
        img = img.astype(np.float32)

        # IMPORTANT: MergeMertens expects float in [0,1]
        img = img / 255.0

        cleaned.append(img)

    # --- FINAL SAFETY CHECK ---
    if len(cleaned) < 2:
        raise ValueError("Not enough valid images after cleanup")

    # --- ENSURE ALL SAME SHAPE (double safety pass) ---
    shapes = [i.shape for i in cleaned]
    if len(set(shapes)) != 1:
        raise ValueError(f"Shape mismatch after preprocessing: {shapes}")

    merge = cv2.createMergeMertens()

    result = merge.process(cleaned)

    return result

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
