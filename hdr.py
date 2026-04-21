import cv2
import numpy as np


# ----------------------------
# EXPOSURE FUSION (CORE HDR)
# ----------------------------
def fuse(images):
    merge = cv2.createMergeMertens()
    return merge.process(images)


# ----------------------------
# BASIC ENHANCEMENT (LIGHTROOM BASE LOOK)
# ----------------------------
def enhance(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


# ----------------------------
# SLIDER-BASED COLOR GRADING
# ----------------------------
def grade(img, vividness=1.2, sky_boost=1.3):

    img = img.astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # general vibrance
    s = np.clip(s * vividness, 0, 255)

    # sky boost (blue hue range)
    sky_mask = (h > 90) & (h < 140)
    s[sky_mask] = np.clip(s[sky_mask] * sky_boost, 0, 255)

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def process(images, vividness=1.2, sky_boost=1.3):

    if images is None or len(images) == 0:
        raise ValueError("No images provided")

    images = [img for img in images if img is not None]

    if len(images) < 2:
        raise ValueError("Need at least 2 images")

    # HDR fusion
    fused = fuse(images)

    if fused is None:
        raise ValueError("Fusion failed")

    # convert float HDR → 8-bit
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # Lightroom-style tone shaping
    fused = enhance(fused)

    # final grading (sliders)
    fused = grade(fused, vividness, sky_boost)

    return fused
