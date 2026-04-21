import cv2
import numpy as np


# ----------------------------
# SAFE NORMALIZATION
# ----------------------------
def normalize(images):

    cleaned = []

    base = images[0]
    h, w = base.shape[:2]

    for img in images:

        if img is None:
            continue

        # resize
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # force BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        cleaned.append(img.astype(np.float32) / 255.0)

    return cleaned


# ----------------------------
# EXPOSURE FUSION (NUMPY ONLY)
# ----------------------------
def fuse(images):

    imgs = normalize(images)

    if len(imgs) < 2:
        raise ValueError("Need at least 2 images")

    h, w = imgs[0].shape[:2]

    result = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)

    for img in imgs:

        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # good exposure weighting curve
        weight = np.exp(-((gray - 0.5) ** 2) / 0.07)

        weight = weight[..., None]

        result += img * weight
        weight_sum += weight

    result /= np.clip(weight_sum, 1e-6, None)

    return np.clip(result, 0, 1)


# ----------------------------
# TONE + COLOR IMPROVEMENT
# ----------------------------
def enhance(img):

    img = (img * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.2, (8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------
# FINAL PROCESS PIPELINE
# ----------------------------
def process_images(images):

    fused = fuse(images)

    final = enhance(fused)

    return final
