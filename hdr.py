import cv2
import numpy as np


# ----------------------------
# SAFE ALIGNMENT (NO CRASHES)
# ----------------------------
def align(images):
    """
    Cloud-safe version:
    Skip alignment to avoid OpenCV AlignMTB instability on Render.
    Real estate images (tripod/bracketed) are usually already aligned enough.
    """
    return images


# ----------------------------
# EXPOSURE FUSION (CORE HDR)
# ----------------------------
def fuse(images):
    """
    OpenCV Mertens exposure fusion
    Works reliably in cloud environments
    """
    merge = cv2.createMergeMertens()
    return merge.process(images)


# ----------------------------
# REAL ESTATE ENHANCEMENT
# ----------------------------
def enhance(img):

    if img is None:
        return None

    # Convert to LAB for better tonal control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Local contrast boost (prevents flat HDR look)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Mild brightness/contrast tuning (MLS-friendly)
    img = cv2.convertScaleAbs(img, alpha=1.08, beta=8)

    return img


# ----------------------------
# MAIN HDR PIPELINE
# ----------------------------
def process(images):

    # ---------------- VALIDATION ----------------
    if images is None or len(images) == 0:
        raise ValueError("No images provided to HDR pipeline")

    # remove broken images
    images = [img for img in images if img is not None]

    if len(images) < 2:
        raise ValueError("Need at least 2 valid images for HDR")

    # ---------------- ALIGN ----------------
    aligned = align(images)

    # ---------------- FUSE ----------------
    fused = fuse(aligned)

    if fused is None:
        raise ValueError("HDR fusion failed (fused image is None)")

    # ---------------- CONVERT FLOAT → 8-bit ----------------
    fused = np.nan_to_num(fused)
    fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

    # ---------------- ENHANCE ----------------
    final = enhance(fused)

    if final is None:
        raise ValueError("Enhancement failed")

    return final
