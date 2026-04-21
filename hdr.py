import cv2
import numpy as np

def align(images):
    alignMTB = cv2.createAlignMTB()
    aligned = []
    alignMTB.process(images, aligned)
    return aligned

def fuse(images):
    merge = cv2.createMergeMertens()
    return merge.process(images)

def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img

def process(images):
    aligned = align(images)
    fused = fuse(aligned)

    img = np.clip(fused * 255, 0, 255).astype(np.uint8)
    return enhance(img)