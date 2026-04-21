import cv2
import numpy as np


# ----------------------------
# SAFE ALIGNMENT (cloud stable)
# ----------------------------
def align(images):
    # Skip alignment for stability on Render
    # (most bracketed real estate shots are tripod-based anyway)
    return images


# ----------------------------
# EXPOSURE FUSION (BASE HDR)
# ----------------------------
def fuse(images):
    merge = cv2.createMergeMertens()
    return merge.process(images)


# ----------------------------
# LIGHTROOM-STYLE TONE MAPPING
# ----------------------------
def tone_map(img):

    # Convert to float for proper grading
    img = img.astype(np.float32) / 255.0

    # ---- global exposure compression (prevents blown windows)
    img = np.clip(img, 0, 1)

    # soft highlight rolloff (critical “Lightroom look” step)
    highlight_mask = np.power(img, 0.85)

    img = 0.6 * img + 0.4 * highlight_mask

    # ---- gamma correction (interior brightness control)
    gamma = 1.1
    img = np.power(img, 1.0 / gamma)

    return img


# ----------------------------
# LOCAL CONTRAST (CLARITY-LIKE)
# ----------------------------
def local_contrast(img):

    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img.astype(np.float32) / 255.0


# ----------------------------
# FINAL COLOR GRADING
# ----------------------------
def grade(img):

    img = (img * 255).astype(np.uint8)

    # mild saturation boost (real estate pop)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = cv2.add(s, 8)  # subtle saturation boost

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def process(images):

    if images is None or len(images) == 0:
        raise ValueError("No images provided")

    images = [img for img in images if img is not None]

    if len(images) < 2:
        raise ValueError("Need at least 2 images")

    # 1. align (safe)
    aligned = align(images)

    # 2. exposure fusion (HDR base)
    fused = fuse(aligned)

    if fused is None:
        raise ValueError("Fusion failed")

    # 3. convert float HDR → 8-bit
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # 4. tone mapping (LIGHTROOM-LIKE BEHAVIOR)
    toned = tone_map(fused)

    # 5. local contrast (clarity)
    contrasted = local_contrast(toned)

    # 6. final grading
    final = grade(contrasted)

    return final
